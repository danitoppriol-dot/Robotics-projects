#!/usr/bin/env python3

from enum import Enum
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist        
from std_msgs.msg import Bool               
from irob_interfaces.srv import Activate, Deactivate, GetGoal, AtGoal


# BEHAVIOR TREE NODE STATUSES
#ogni bt puo restituire questi valori 
class NodeStatus(Enum):
    SUCCESS = 0
    FAILURE = 1
    RUNNING = 2

# BASE BEHAVIOR TREE NODE - tutti ereditano da questa
class BTNode:
    def __init__(self, name):
        self.name = name
        self.status = NodeStatus.FAILURE
    
    def tick(self): #ritorna lo stato sopra
        raise NotImplementedError


# COMPOSITE NODES - nodi di condizione azione quelli senza figli sono foglie gli altri rami
class SequenceNode(BTNode):
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children
        self.current_child = 0
    
    def tick(self):
        while self.current_child < len(self.children):
            child_status = self.children[self.current_child].tick()
            
            if child_status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status
            elif child_status == NodeStatus.FAILURE:
                self.current_child = 0
                self.status = NodeStatus.FAILURE
                return self.status
            else:  # SUCCESS
                self.current_child += 1
        
        self.current_child = 0
        self.status = NodeStatus.SUCCESS
        return self.status


class FallbackNode(BTNode):
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children
        self.current_child = 0
    
    def tick(self):
        while self.current_child < len(self.children):
            child_status = self.children[self.current_child].tick()
            
            if (child_status == NodeStatus.RUNNING):
                self.status = NodeStatus.RUNNING
                return self.status
            elif (child_status == NodeStatus.SUCCESS):
                self.current_child = 0
                self.status = NodeStatus.SUCCESS
                return self.status
            else:  # FAILURE
                self.current_child += 1
        
        self.current_child = 0
        self.status = NodeStatus.FAILURE
        return self.status
    

# DECISION TREE CONDITION NODES 

class CheckRobotInactive(BTNode):
    def __init__(self, robot_node):
        super().__init__("CheckRobotInactiveFirstTime")
        self.robot_node = robot_node
        self.first_activation = True  # Flag per permettere la prima attivazione automatica
    
    def tick(self):
        self.robot_node.get_logger().info("CHECK THE ROBOT INACTIVE")
        
        # Permetti l'attivazione se:
        # 1. È la prima volta (avvio automatico)
        # 2. OPPURE c'è una richiesta esterna di attivazione
        if not self.robot_node.is_robot_active and not self.robot_node.missionComplete:
            if self.first_activation or self.robot_node.external_activation:
                self.robot_node.get_logger().info("Robot is inactive - ready for activation")
                if self.first_activation:
                    self.robot_node.get_logger().info("First activation - starting automatically")
                    self.first_activation = False
                self.status = NodeStatus.SUCCESS
            else:
                self.status = NodeStatus.FAILURE
        else:
            self.status = NodeStatus.FAILURE
        return self.status

class CheckRobotLost(BTNode): # controllo che non sa più dove si trovi
    # CAPISCI SE IL ROBOT NON SI TROVA PIU
    # per fare questo prendo amcl e guardo la covarianza se questa e maggiore di 1 allora il robot si e perso 
    def __init__(self, robot_node):
        super().__init__("CheckRobotLost")
        self.robot_node = robot_node
    
    def tick(self):
        self.robot_node.get_logger().info("CHECK ROBOT LOST")
        
        # Se il robot non è attivo, non ha senso controllare se è perso
        if not self.robot_node.is_robot_active:
            self.robot_node.get_logger().info("CHECK ROBOT LOST - Robot inactive")
            self.status = NodeStatus.FAILURE
            return self.status
        
        # PRIORITÀ 1: Se il flag robotLost è già stato impostato (da Navigation o altro)
        if self.robot_node.robotLost:
            self.robot_node.get_logger().warn("CHECK ROBOT LOST - Flag robotLost già attivo!")
            self.status = NodeStatus.SUCCESS
            return self.status
        
        # PRIORITÀ 2: All'inizio, il robot deve localizzarsi
        if self.robot_node.robot_pose is None:
            self.robot_node.get_logger().info("CHECK ROBOT LOST - no robot position (first localization)")
            self.robot_node.robotLost = True
            self.status = NodeStatus.SUCCESS
            return self.status
        
        # PRIORITÀ 3: Se non abbiamo dati di covarianza, aspetta
        if (self.robot_node.cov_xx is None or self.robot_node.cov_xy is None or 
            self.robot_node.cov_yx is None or self.robot_node.cov_yy is None or 
            self.robot_node.cov_theta is None):
            self.status = NodeStatus.RUNNING
            return self.status
        
        # PRIORITÀ 4: Controlla la covarianza
        pos_cov = np.array([
            [self.robot_node.cov_xx, self.robot_node.cov_xy], 
            [self.robot_node.cov_yx, self.robot_node.cov_yy]
        ])

        max_eig = np.max(np.linalg.eigvals(pos_cov))
        yaw_var = self.robot_node.cov_theta

        if max_eig > 1.0 or yaw_var > 0.5:
            self.robot_node.get_logger().warn(
                f"Localizzazione persa! max_eig={max_eig:.3f}, yaw_var={yaw_var:.3f}"
            )
            self.robot_node.robotLost = True
            self.status = NodeStatus.SUCCESS
        else:
            self.robot_node.get_logger().info("Localizzazione stabile.")
            self.status = NodeStatus.FAILURE
        
        return self.status

class CheckRobotActive(BTNode):
    #CONTROLLO CHE IL ROBOT SIA ATTIVO PRIMA DI INIZIARE UNA MISSIONE
    def __init__(self, robot_node):
        super().__init__("CheckRobotActive")
        self.robot_node = robot_node
    
    def tick(self):
        self.robot_node.get_logger().info("CHECK ROBOT ACTIVE")
        
        if self.robot_node.is_robot_active:
            self.status = NodeStatus.SUCCESS
        else:
            self.robot_node.get_logger().info("Robot is INACTIVE - skipping mission logic")
            self.status = NodeStatus.FAILURE
        
        return self.status

class CheckGoalAvailable(BTNode):
    # controllare che il goal che viene dato non sia in un ostacolo 
    # prendi lista e controlla che la posizione non abbia come valore 100 -> Navigation
    def __init__(self, robot_node):
        super().__init__("CheckGoalAvailability")
        self.robot_node = robot_node  

    def tick(self):
        self.robot_node.get_logger().info("CHECK GOAL AVAILABLE")
            
        # CONTROLLO PRIORITARIO: Se il robot è perso, interrompi
        if self.robot_node.robotLost:
            self.robot_node.get_logger().warn("CheckGoalAvailable: Robot is lost, aborting")
            self.status = NodeStatus.FAILURE
            return self.status
        
        if self.robot_node.external_deactivation:
            self.robot_node.get_logger().warn("CheckGoalAvailable: Disattivazione esterna rilevata!")
            self.status = NodeStatus.FAILURE
            return self.status
        
        if self.robot_node.goalValidity and self.robot_node.arrived_goal is not None:
            self.robot_node.get_logger().info(
                f"CheckGoalAvailable: Goal già validato: {self.robot_node.arrived_goal}"
            )
            self.status = NodeStatus.SUCCESS
            return self.status
        
        if self.robot_node.arrived_goal is None:
            self.robot_node.get_logger().warn("CheckGoalAvailable: No goal available yet")
            self.status = NodeStatus.FAILURE
            return self.status
        
        if self.robot_node.map_data is None:
            self.robot_node.get_logger().warn("CheckGoalAvailable: Waiting for map data...")
            self.status = NodeStatus.RUNNING
            return self.status
            
        goal_x, goal_y = self.robot_node.arrived_goal
        if self.robot_node.map_data is None:
            self.robot_node.get_logger().warn("CheckGoalAvailable: Waiting for map data...")
            self.status = NodeStatus.RUNNING
            return self.status

        # Usa sempre i parametri di map_data
        resolution = float(self.robot_node.map_data.info.resolution)
        origin_x = float(self.robot_node.map_data.info.origin.position.x)
        origin_y = float(self.robot_node.map_data.info.origin.position.y)
        width = int(self.robot_node.map_data.info.width)
        height = int(self.robot_node.map_data.info.height)

        # compute world bounds (x_min <= x < x_max_excl)
        x_min = origin_x
        y_min = origin_y
        x_max_excl = x_min + resolution * width
        y_max_excl = y_min + resolution * height

        # quick world-bounds check
        eps = 1e-9
        in_x_bounds = (x_min - eps) <= goal_x < (x_max_excl + eps)
        in_y_bounds = (y_min - eps) <= goal_y < (y_max_excl + eps)

        if not (in_x_bounds and in_y_bounds):
            # either reject or clip depending on parameter
            if getattr(self.robot_node, 'clip_out_of_bounds', False):
                clipped_x = min(max(goal_x, x_min + self.robot_node.clip_margin), x_max_excl - self.robot_node.clip_margin)
                clipped_y = min(max(goal_y, y_min + self.robot_node.clip_margin), y_max_excl - self.robot_node.clip_margin)
                self.robot_node.get_logger().warn(f"Goal ({goal_x:.2f},{goal_y:.2f}) out of map bounds; clipping to ({clipped_x:.2f},{clipped_y:.2f})")
                goal_x, goal_y = clipped_x, clipped_y
                # update arrived_goal so subsequent nodes see clipped coords
                self.robot_node.arrived_goal = (goal_x, goal_y)
            else:
                self.robot_node.get_logger().warn(f"Goal ({goal_x:.2f}, {goal_y:.2f}) is outside map world bounds [{x_min:.2f},{x_max_excl:.2f}]x[{y_min:.2f},{y_max_excl:.2f}]")
                self.robot_node.goalValidity = False
                self.robot_node.arrived_goal = None
                self.status = NodeStatus.FAILURE
                return self.status

        # convert to indices
        col = int((goal_x - origin_x) / resolution)
        row = int((goal_y - origin_y) / resolution)

        # validate indices
        if row < 0 or col < 0 or row >= height or col >= width:
            self.robot_node.get_logger().warn(f"Goal ({goal_x:.2f}, {goal_y:.2f}) maps to out-of-range indices ({row},{col}) for map size ({height},{width})")
            self.robot_node.goalValidity = False
            self.robot_node.arrived_goal = None
            self.status = NodeStatus.FAILURE
            return self.status

        # read occupancy value from authoritative source
        idx = col + row * width
        try:
            cell_value = self.robot_node.map_data.data[idx]
        except Exception as e:
            self.robot_node.get_logger().warn(f"Failed to read cell from map_data.data: {e}")
            self.robot_node.goalValidity = False
            self.robot_node.arrived_goal = None
            self.status = NodeStatus.FAILURE
            return self.status

        if cell_value == 100 or cell_value == -1:
            self.robot_node.get_logger().info(f"Goal cell value: {cell_value} not reacheable")
            self.robot_node.goalValidity = False
            self.robot_node.arrived_goal = None
            self.status = NodeStatus.FAILURE
        else:
            self.robot_node.get_logger().info(f"Goal cell value: {cell_value} reacheable")
            #ho gia impostato l'arrived_goal
            self.robot_node.goalValidity = True
            self.status = NodeStatus.SUCCESS

        self.robot_node.get_logger().info(f"Goal ({goal_x:.2f}, {goal_y:.2f}) -> grid ({row}, {col}) -> cell value: {cell_value}")
           
        return self.status
    
# class CheckExternalGoalRequested(BTNode):
#     def __init__(self, robot_node):
#         super().__init__("CheckExternalGoalRequested")
#         self.robot_node = robot_node
    
#     def tick(self):
#         self.robot_node.get_logger().info(" X - CHECK EXTERNAL GOAL REQUEST")
#         if self.robot_node.external_goal_request:
#             self.robot_node.get_logger().info("External goal request detected")
#             self.robot_node.stop_robot()  # Ferma la navigazione corrente
#             self.robot_node.current_goal = None  # Resetta per forzare nuovo goal
#             self.robot_node.external_goal_request = False
#             self.status = NodeStatus.SUCCESS
#         else:
#             self.status = NodeStatus.FAILURE
#         return self.status
    

class CheckAtGoal(BTNode):
    # controlla con il servizio se il goal e arrivato 
    def __init__(self, robot_node):
        super().__init__("CheckAtGoal")
        self.robot_node = robot_node
        self.at_goal_future = None
        self.creeping = False
        self.waiting_for_service = False # Flag per sapere se stiamo aspettando una risposta

    def tick(self):
        self.robot_node.get_logger().info("CHECK AT GOAL")

        # --- FASE 1: Stiamo aspettando una risposta dal servizio? ---
        if self.robot_node.is_robot_lost():
            self.robot_node.get_logger().warn("ROBOT PERSO! Interrompo...")
            self.robot_node.stop_robot()
            self.robot_node.robotLost = True
            # Salva il goal per riprenderlo dopo la rilocalizzazione
            if self.robot_node.current_goal is not None:
                self.robot_node.saved_goal_after_lost = self.robot_node.current_goal
            self.status = NodeStatus.FAILURE
            return self.status
        
        if self.robot_node.external_deactivation:
            self.robot_node.get_logger().warn("CheckAtGoal: Disattivazione esterna rilevata!")
            self.robot_node.stop_robot()
            self.creeping = False
            self.at_goal_future = None
            self.waiting_for_service = False
            self.status = NodeStatus.FAILURE
            return self.status
        
        if self.at_goal_future:
            if self.at_goal_future.done():
                self.waiting_for_service = False
                try:
                    result = self.at_goal_future.result()
                    self.at_goal_future = None # Resetta il future
                    
                    if result.success:
                        # ***** SUCCESSO FINALE! *****
                        self.robot_node.get_logger().info("CheckAtGoal: Goal RAGGIUNTO! (confermato da servizio)")
                        self.creeping = False
                        
                        # Resetta tutto lo stato per il prossimo goal
                        self.robot_node.navigating = False
                        self.robot_node.arrivedGoalNav2 = False
                        self.robot_node.previous_goal = self.robot_node.current_goal
                        self.robot_node.current_goal = None
                        self.robot_node.arrived_goal = None
                        self.robot_node.goalValidity = False
                        self.robot_node.navigationSafe = False
                        
                        self.status = NodeStatus.SUCCESS
                        return self.status
                    else:
                        # Servizio dice NO, avvia il "creep"
                        self.robot_node.get_logger().warn("Servizio 'at_goal' dice NO. Avvio creep.")
                        self.creeping = True
                        self.status = NodeStatus.RUNNING # Rimani in questo nodo per il creep
                        return self.status

                except Exception as e:
                    self.robot_node.get_logger().error(f"Errore servizio 'at_goal': {e}")
                    self.at_goal_future = None
                    self.status = NodeStatus.FAILURE # Il servizio è fallito
                    return self.status
            else:
                # Il servizio è stato chiamato, ma non ha ancora risposto
                self.status = NodeStatus.RUNNING
                return self.status

        # --- FASE 2: Stiamo "creeping"? ---
        if self.creeping:
            goal_x, goal_y = self.robot_node.current_goal
            arrived = self.robot_node.creep_to_goal(goal_x, goal_y, tolerance=0.15)
            if arrived:
                self.robot_node.get_logger().info("Creep completato! Richiamo servizio 'at_goal'.")
                self.creeping = False # Smetti di muoverti
                # Al prossimo tick, ricadremo nella FASE 3 per richiamare il servizio
            
            self.status = NodeStatus.RUNNING # Continua a fare creep
            return self.status
        
        # --- FASE 3: Nav2 ha finito, dobbiamo chiamare il servizio? ---
        # (Siamo qui solo se future=None e creeping=False)
        if self.robot_node.arrivedGoalNav2 and not self.waiting_for_service:
            if self.robot_node.at_goal_client.wait_for_service(timeout_sec=0.5):
                self.robot_node.get_logger().info("CheckAtGoal: Chiamo servizio 'at_goal'...")
                self.at_goal_future = self.robot_node.at_goal_client.call_async(AtGoal.Request())
                self.waiting_for_service = True
                self.status = NodeStatus.RUNNING
                return self.status
            else:
                self.robot_node.get_logger().warn("At_goal service not available")
                self.status = NodeStatus.FAILURE
                return self.status

        # --- FASE 4: Se siamo qui, Nav2 non ha ancora finito. ---
        # (Questo stato non dovrebbe essere raggiunto se Navigation.tick() è corretto,
        # ma è una sicurezza)
        if self.robot_node.navigating:
            self.status = NodeStatus.RUNNING
            return self.status

        # Se non siamo in nessuno stato valido, fallisci
        self.status = NodeStatus.FAILURE
        return self.status


# DECISION TREE ACTION NODES
class Activation(BTNode):
    def __init__(self, robot_node):
        super().__init__("Activation")
        self.robot_node = robot_node
        self.activation_sent = False
        self.service_future = None

    def tick(self):
        self.robot_node.get_logger().info("HANDLE ACTIVATION")
        
        if self.robot_node.is_robot_active and not self.robot_node.external_activation:
            self.status = NodeStatus.FAILURE
            return self.status
        
        if not self.activation_sent:
            self.robot_node.get_logger().info("Activating robot...")
            if self.robot_node.activate_client.wait_for_service(timeout_sec=1.0):
                self.service_future = self.robot_node.activate_client.call_async(Activate.Request())
                self.activation_sent = True
                self.status = NodeStatus.RUNNING
            else:
                self.robot_node.get_logger().warn("Activate service not available")
                self.status = NodeStatus.FAILURE
        else:
            if self.service_future.done():
                try:
                    result = self.service_future.result()
                    if result.success:
                        self.robot_node.get_logger().info("Robot successfully activated.")
                        self.robot_node.is_robot_active = True
                        
                        # GESTIONE RIATTIVAZIONE
                        if self.robot_node.external_activation:
                            self.robot_node.get_logger().info("=== HANDLING REACTIVATION ===")
                            self.robot_node.external_activation = False
                            
                            if self.robot_node.saved_goal_before_deactivation is not None:
                                goal = self.robot_node.saved_goal_before_deactivation
                                self.robot_node.get_logger().info(f"Ripristino goal salvato: {goal}")
                                
                                # IMPOSTA arrived_goal (non current_goal!)
                                # Così il BT passa per CheckGoalAvailable e PrepareNavigation
                                self.robot_node.arrived_goal = goal
                                self.robot_node.saved_goal_before_deactivation = None
                                
                                # Reset flags per permettere la validazione
                                self.robot_node.goalValidity = False  # CheckGoalAvailable lo ricontrollerà
                                self.robot_node.navigationSafe = True
                                self.robot_node.navigating = False
                                self.robot_node.arrivedGoalNav2 = False
                            else:
                                self.robot_node.get_logger().warn("Nessun goal salvato da ripristinare")
                        
                        self.status = NodeStatus.SUCCESS
                    else:
                        self.robot_node.get_logger().warn("Activation failed.")
                        self.status = NodeStatus.FAILURE
                except Exception as e:
                    self.robot_node.get_logger().error(f"Activation service error: {e}")
                    self.status = NodeStatus.FAILURE
                finally:
                    self.activation_sent = False
            else:
                self.status = NodeStatus.RUNNING

        return self.status

class HandleRelocalization(BTNode):
    # gestisci la rilocalizzazione PARTICLE FILTER
    # mi rendo conto cge si e perso dal check robot lost  quindi fermo nav2
    # mi salvo il goal corrente e riinizio con la localizzazione 
    # inizio a fare un movimento circolare (ho bisogno di twist) con velocità lineare 0 e angolare piccola
    # quando la covarianza torna sotto una soglia vuol dire che mi sono ritrovato 
    # quindi fermo twisti, e riprendo la navigazione al goal salvato
    def __init__(self, robot_node):
        super().__init__("HandleRelocalization")
        self.robot_node = robot_node
        self.savedGoal = None
        self.relocalization_started = False
    
    def tick(self):
        self.robot_node.get_logger().info("HANDLE RELOCALIZATION")
        
        # Prima volta che entriamo qui: ferma tutto e salva il goal
        if not self.relocalization_started:
            self.robot_node.get_logger().info("Inizio rilocalizzazione - fermo navigazione")
            self.robot_node.stop_robot()
            
            # Salva il goal corrente (potrebbe venire da Navigation o essere già salvato)
            if hasattr(self.robot_node, 'saved_goal_after_lost') and self.robot_node.saved_goal_after_lost:
                self.savedGoal = self.robot_node.saved_goal_after_lost
            elif self.robot_node.current_goal is not None:
                self.savedGoal = self.robot_node.current_goal
            
            if self.savedGoal:
                self.robot_node.get_logger().info(f"Goal salvato per dopo: {self.savedGoal}")
            
            self.relocalization_started = True
        
        # Movimento circolare per aiutare AMCL
        self.robot_node.get_logger().info("Movimento circolare per rilocalizzazione...")
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.5  # Velocità angolare
        self.robot_node.cmd_vel_pub.publish(twist_msg)

        # Controlla la covarianza
        if (self.robot_node.cov_xx is None or self.robot_node.cov_xy is None or 
            self.robot_node.cov_yx is None or self.robot_node.cov_yy is None or 
            self.robot_node.cov_theta is None):
            self.status = NodeStatus.RUNNING
            return self.status
        
        pos_cov = np.array([
            [self.robot_node.cov_xx, self.robot_node.cov_xy], 
            [self.robot_node.cov_yx, self.robot_node.cov_yy]
        ])

        max_eig = np.max(np.linalg.eigvals(pos_cov))
        yaw_var = self.robot_node.cov_theta
        
        # Soglia per considerarsi rilocalizzati
        if max_eig < 1.0 and yaw_var < 0.5:
            self.robot_node.get_logger().info("Rilocalizzazione completata!")
            
            # Ferma il movimento circolare
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.robot_node.cmd_vel_pub.publish(stop_msg)
            
            # Riprendi la navigazione verso il goal salvato
            if self.savedGoal is not None:
                x, y = self.savedGoal
                self.robot_node.get_logger().info(f"Riprendo navigazione verso: ({x:.2f}, {y:.2f})")
                self.robot_node.current_goal = self.savedGoal
                self.robot_node.navigate_to(x, y)
            
            # Reset flags
            self.robot_node.robotLost = False
            self.robot_node.saved_goal_after_lost = None
            self.relocalization_started = False
            self.savedGoal = None
            
            self.status = NodeStatus.SUCCESS
        else:
            self.robot_node.get_logger().info(f"Rilocalizzazione in corso... (eig={max_eig:.3f}, yaw={yaw_var:.3f})")
            self.status = NodeStatus.RUNNING

        return self.status

class GetNewGoal(BTNode):
    def __init__(self, robot_node):
        super().__init__("GetNewGoal")
        self.robot_node = robot_node
        self.service_future = None
        self.waiting_for_response = False

    def tick(self):
        self.robot_node.get_logger().info("GET NEW GOAL")

        if self.robot_node.robotLost:
            self.robot_node.get_logger().warn("GetNewGoal: Robot is lost, skipping goal request")
            self.waiting_for_response = False
            self.service_future = None
            self.status = NodeStatus.FAILURE
            return self.status
        
        if self.robot_node.external_deactivation:
            self.robot_node.get_logger().warn("GetNewGoal: Disattivazione esterna rilevata!")
            self.waiting_for_response = False
            self.service_future = None
            self.status = NodeStatus.FAILURE
            return self.status
         
        # PRIORITÀ 1: Se c'è un goal salvato da riattivazione, ripristinalo PRIMA
        if self.robot_node.saved_goal_before_deactivation is not None:
            self.robot_node.get_logger().info(
                f"GetNewGoal: Ripristino goal salvato da disattivazione: {self.robot_node.saved_goal_before_deactivation}"
            )
            self.robot_node.arrived_goal = self.robot_node.saved_goal_before_deactivation
            self.robot_node.saved_goal_before_deactivation = None
            self.robot_node.goalValidity = False  # Forza la rivalidazione
            self.status = NodeStatus.SUCCESS
            return self.status
        
        # PRIORITÀ 2: Se c'è già un arrived_goal (da altre fonti), non chiederne uno nuovo
        if self.robot_node.arrived_goal is not None:
            self.robot_node.get_logger().info(
                f"GetNewGoal: Goal già presente: {self.robot_node.arrived_goal}"
            )
            self.status = NodeStatus.SUCCESS
            return self.status
        
        # Se non stiamo aspettando una risposta, chiamiamo il servizio
        if not self.waiting_for_response:
            self.robot_node.get_logger().info("Requesting new goal...")
            if self.robot_node.get_goal_client.wait_for_service(timeout_sec=1.0):
                self.service_future = self.robot_node.get_goal_client.call_async(GetGoal.Request())
                self.waiting_for_response = True
                self.status = NodeStatus.RUNNING
            else:
                self.robot_node.get_logger().warn("Get_goal service not available")
                self.status = NodeStatus.FAILURE
            return self.status
        
        # Se stiamo aspettando una risposta, controlliamo se è arrivata
        if self.service_future and self.service_future.done():
            self.waiting_for_response = False
            try:
                response = self.service_future.result()
                new_goal = (response.goal_x, response.goal_y)
                
                # Confronta il NUOVO goal con il VECCHIO goal
                if new_goal == self.robot_node.previous_goal:
                    self.robot_node.get_logger().info(
                        f"Ricevuto goal {new_goal}, uguale al precedente. Missione completata!"
                    )
                    self.robot_node.missionComplete = True
                    self.robot_node.arrived_goal = None
                    self.status = NodeStatus.FAILURE
                else:
                    self.robot_node.get_logger().info(
                        f"New goal received: {new_goal}, previous {self.robot_node.current_goal}"
                    )
                    self.robot_node.arrived_goal = new_goal
                    self.robot_node.goalValidity = False
                    self.robot_node.previous_goal = new_goal
                    self.status = NodeStatus.SUCCESS
                    
            except Exception as e:
                self.robot_node.get_logger().error(f"Failed to get goal: {e}")
                self.status = NodeStatus.FAILURE
            finally:
                self.service_future = None
            
            return self.status
        
        # Se non siamo in nessuno dei casi, stiamo ancora aspettando
        self.status = NodeStatus.RUNNING
        return self.status
    
class PrepareNavigation(BTNode):
    # setta il goal corrente LO FACCIO QUA PERCHÈ SO CHE IL CHECK DEL GOAL 
    # È ANDATO A BUON FINE QUIDNI POSSO INIZIARE LA NAVIGAZIONE
    def __init__(self, robot_node):
        super().__init__("PrepareNavigation")
        self.robot_node = robot_node
    
    def tick(self):
        self.robot_node.get_logger().info("PREPARE NAVIGATION")
        
        # CONTROLLO PRIORITARIO: Se il robot è perso, interrompi
        if self.robot_node.robotLost:
            self.robot_node.get_logger().warn("PrepareNavigation: Robot is lost, aborting")
            self.status = NodeStatus.FAILURE
            return self.status
        
        if self.robot_node.external_deactivation:
            self.robot_node.get_logger().warn("PrepareNavigation: Disattivazione esterna rilevata!")
            self.status = NodeStatus.FAILURE
            return self.status
        
        if self.robot_node.robot_pose is None:
            self.robot_node.get_logger().warn("Robot pose is None - not localized!")
            self.status = NodeStatus.FAILURE
            return self.status
        
        # Prepara la navigazione
        self.robot_node.get_logger().info("Preparing for navigation to new goal")
        
        # Setta il current goal
        self.robot_node.current_goal = self.robot_node.arrived_goal
        self.robot_node.arrived_goal = None
        self.robot_node.arrivedGoalNav2 = False 
        
        # AVVIA LA NAVIGAZIONE
        x, y = self.robot_node.current_goal
        success = self.robot_node.navigate_to(x, y)
        
        if success:
            self.robot_node.navigationSafe = True
            self.robot_node.get_logger().info("Navigation started successfully")
            self.status = NodeStatus.SUCCESS
        else:
            self.robot_node.get_logger().error("Failed to start navigation!")
            self.robot_node.current_goal = None
            self.robot_node.navigationSafe = False
            self.status = NodeStatus.FAILURE
                
        return self.status


class Navigation(BTNode): 
    def __init__(self, robot_node):
        super().__init__("Navigation")
        self.robot_node = robot_node

    def tick(self):
        self.robot_node.get_logger().info("NAVIGATION")
        
        # 0. PRIORITÀ MASSIMA: Controlla disattivazione esterna
        if self.robot_node.external_deactivation:
            self.robot_node.get_logger().warn("DISATTIVAZIONE ESTERNA DURANTE NAVIGAZIONE! Interrompo.")
            self.status = NodeStatus.FAILURE
            return self.status
        
        # 1. PRIORITÀ: Controlla se il robot si è perso
        if self.robot_node.is_robot_lost():
            self.robot_node.get_logger().warn("ROBOT PERSO! Interrompo navigazione...")
            self.robot_node.stop_robot()
            self.robot_node.robotLost = True
            if self.robot_node.current_goal is not None:
                self.robot_node.saved_goal_after_lost = self.robot_node.current_goal
            
            self.robot_node.navigating = False
            self.robot_node.arrivedGoalNav2 = False
            
            self.status = NodeStatus.FAILURE
            return self.status
        
        # 2. Se non c'è un goal, fallisci
        if self.robot_node.current_goal is None:
            self.robot_node.get_logger().warn("Navigation: No current goal.")
            self.status = NodeStatus.FAILURE
            return self.status
        
        # 3. Se Nav2 ha finito, SUCCESS per passare a CheckAtGoal
        if self.robot_node.arrivedGoalNav2:
            self.robot_node.get_logger().info("Navigation: Nav2 completato. → CheckAtGoal")
            self.status = NodeStatus.SUCCESS
            return self.status

        # 4. Se stiamo navigando, RUNNING
        if self.robot_node.navigating:
            self.status = NodeStatus.RUNNING
            return self.status
        
        # 5. Altrimenti fallisci
        self.robot_node.get_logger().warn("Navigation: Non in navigazione e Nav2 non completato.")
        self.status = NodeStatus.FAILURE
        return self.status


class HandleDeactivation(BTNode):
    def __init__(self, robot_node):
        super().__init__("Deactivation")
        self.robot_node = robot_node
        self.deactivation_sent = False
        self.service_future = None

    def tick(self):
        self.robot_node.get_logger().info("HANDLE DEACTIVATION")

        # Verifica se è richiesta una disattivazione
        if self.robot_node.missionComplete or self.robot_node.external_deactivation:
            # === PRIMA CHIAMATA AL SERVIZIO ===
            if not self.deactivation_sent:
                self.robot_node.get_logger().info("Deactivating robot...")

                # Ferma eventuale navigazione
                if self.robot_node.navigating:
                    self.robot_node.stop_robot()
                    self.robot_node.current_goal = None
                    self.robot_node.arrived_goal = None
                    self.robot_node.navigating = False
                    self.robot_node.arrivedGoalNav2 = False

                # Prova a chiamare il servizio di disattivazione
                if self.robot_node.deactivate_client.wait_for_service(timeout_sec=1.0):
                    self.service_future = self.robot_node.deactivate_client.call_async(Deactivate.Request())
                    self.deactivation_sent = True
                    self.status = NodeStatus.RUNNING
                else:
                    self.robot_node.get_logger().warn("Deactivate service not available - disattivazione locale")
                    # Fallback locale
                    self.robot_node.is_robot_active = False
                    self.robot_node.external_deactivation = False
                    self.deactivation_sent = False
                    self.status = NodeStatus.SUCCESS

            # === ASPETTA LA RISPOSTA DEL SERVIZIO ===
            else:
                if self.service_future.done():
                    try:
                        result = self.service_future.result()
                        if result.success:
                            self.robot_node.get_logger().info("Robot successfully deactivated.")
                            self.robot_node.is_robot_active = False

                            if self.robot_node.missionComplete:
                                self.robot_node.get_logger().info("MISSIONE COMPLETATA - Robot disattivato definitivamente")
                                self.robot_node.mission_ended = True
                            else:
                                self.robot_node.get_logger().info("Disattivazione esterna completata")

                            self.robot_node.external_deactivation = False
                            self.deactivation_sent = False
                            self.service_future = None

                            self.status = NodeStatus.SUCCESS
                        else:
                            self.robot_node.get_logger().warn("Deactivate service failed - forcing local deactivation")
                            self.robot_node.is_robot_active = False
                            self.robot_node.external_deactivation = False
                            self.deactivation_sent = False
                            self.service_future = None
                            self.status = NodeStatus.SUCCESS

                    except Exception as e:
                        self.robot_node.get_logger().error(f"Deactivate service error: {e}")
                        self.robot_node.is_robot_active = False
                        self.robot_node.external_deactivation = False
                        self.deactivation_sent = False
                        self.service_future = None
                        self.status = NodeStatus.SUCCESS
                else:
                    self.status = NodeStatus.RUNNING

        else:
            self.status = NodeStatus.FAILURE

        return self.status



# MAIN ROBOT NODE WITH BT
class BTStudentsNode(Node):
    def __init__(self):
        super().__init__('BT_students_nodeA')

        # --- State ---
        self.robot_pose = None
        self.current_goal = None
        self.arrived_goal = None # questo è quello che mi arriva se poi è corretto allora diventa current_goal
        self.previous_goal = None
        self.is_robot_active = False  #il nodo deve partire inattivo 
        self.future = None
        self.navigating = False
        self.map_data = None  # Per la mappa di occupazione
        self.grid = None  # Rappresentazione della mappa come lista 2D
        self.goalValidity = False
        self.navigationSafe = False # ho un goal valido e posso navigare
        self.arrivedGoalNav2 = False
        self.missionComplete = False
        self.cov_xx = None
        self.cov_xy = None
        self.cov_yx = None
        self.cov_yy = None
        self.cov_theta = None
        self.robotLost = False
        self.goal_handle = None  # store the current action goal handle so it can be cancelled
        self.clip_out_of_bounds = False
        self.clip_margin = 0.01
        self.saved_goal_after_lost = None
        self.mission_ended = False 

    
        # --- External Interruption Flags ---
        self.external_goal_request = False
        self.saved_goal_before_deactivation = None
        self.external_activation = False
        self.external_deactivation = False

        # --- Setup Services and Topics ---

        #servizi definiti nell'assigminent client 
        self.activate_client = self.create_client(Activate, 'activate')
        self.deactivate_client = self.create_client(Deactivate, 'deactivate')
        self.get_goal_client = self.create_client(GetGoal, 'get_goal')
        self.at_goal_client = self.create_client(AtGoal, 'at_goal')
        #servizi definiti dell'assigniment server
        self.activate_srv = self.create_service(Activate, 'activate', self.handle_activate_robot)
        self.deactivate_srv = self.create_service(Deactivate, 'deactivate', self.handle_deactivate_robot)
        # servizio per la richiesta di gestione del goal esterno 
        self.get_goal_srv = self.create_service(GetGoal, 'get_goal_external', self.handle_external_goal_request)
        # cliente per la navigazione 
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # topic, mi basta amcl e map per sapere dove sono per la localizzione e robot active per lo stato
        self.pose_subscription = self.create_subscription( PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.robot_state_sub = self.create_subscription(Bool, '/robot_active', self.robot_state_callback, 10)
        map_qos = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_subscription = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        # cmd_vel mi serve per la rilocalizzazione
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- BUILD BEHAVIOR TREE ---
        self.root = self.build_behavior_tree()
        
        self.timer = self.create_timer(0.1, self.bt_tick)

#costruzione del behaviour tree  in ordine di priorita dal maggiore al minore
    def build_behavior_tree(self):
        root = FallbackNode("Root_Priority_Selector", [

                # --- DISATTIVAZIONE ---
                # Se c'è una richiesta di disattivazione, gestiscila e ignora tutto il resto.
                HandleDeactivation(self),

                # --- ATTIVAZIONE ---
                # Se non siamo attivi, questo è l'unico ramo che può avere successo.
                SequenceNode("Initial_Activation_Sequence", [
                    CheckRobotInactive(self),
                    Activation(self)
                ]),

                # --- Rilocalizzazione ---
                # Se siamo attivi ma persi, fermati e rilocalizzati prima di fare qualsiasi altra cosa.
                SequenceNode("Handle_Relocalization_Sequence", [
                    CheckRobotLost(self),
                    HandleRelocalization(self)
                ]),

                # --- LOGICA DI MISSIONE ---
                # Prima controlla che siamo ATTIVI, poi esegui la missione
                SequenceNode("Active_Mission_Guard", [
                    CheckRobotActive(self),  # controlla che siamo ATTIVI
                    FallbackNode("Mission_Logic_Fallback", [
                        #CheckExternalGoalRequested(self),
                        
                        # 4a. Prova a NAVIGARE
                        SequenceNode("Navigation_Sequence", [
                            Navigation(self),
                            CheckAtGoal(self),
                        ]),
                        
                        # 4b. Se la navigazione fallisce, prendi un nuovo goal
                        SequenceNode("Get_New_Goal_Sequence", [
                            GetNewGoal(self),
                            CheckGoalAvailable(self),
                            PrepareNavigation(self)
                        ])
                    ])
                ])
            ])
        return root


    def bt_tick(self): #esegue il nodo, se il principale ritorna success si ferma il robot
    # Se la missione è completamente terminata, ferma il timer
        if self.mission_ended:
            self.get_logger().info("Mission ended - stopping behavior tree")
            self.timer.cancel()
            return
        
        status = self.root.tick()
        
        # Il flag missionComplete viene gestito da HandleDeactivation
        # Non serve fare nulla qui
        
        if status == NodeStatus.FAILURE:
            self.get_logger().debug("Behavior tree returned FAILURE for this tick; continuing to tick.")


    # CALLBACK METHODS 

    def handle_external_goal_request(self, request, response):
        self.get_logger().info("RICHIESTA OBIETTIVO ESTERNA RICEVUTA!")
        
        # Ferma subito il robot
        self.stop_robot()
        self.future = None
        self.external_goal_request = True
        self.get_logger().info("   -> Interruzione azione corrente. Inizio richiesta nuovo obiettivo.")
        
        # Resetta lo stato di navigazione per forzare un nuovo goal
        self.arrived_goal = None
        self.goalValidity = False
        self.current_goal = None

        # Il servizio GetGoal richiede due float come risposta
        response.goal_x = 0.0
        response.goal_y = 0.0

        return response



    def robot_state_callback(self, msg):
        #prendo lo stato del robot dal topic
        self.is_robot_active = msg.data

    def map_callback(self, msg):
        self.map_data = msg
            
            # Log delle informazioni della mappa quando arriva
        mres = float(msg.info.resolution)
        mox = float(msg.info.origin.position.x)
        moy = float(msg.info.origin.position.y)
        mwidth = int(msg.info.width)
        mheight = int(msg.info.height)
            
        x_min = mox
        y_min = moy
        x_max = x_min + mres * mwidth
        y_max = y_min + mres * mheight
            
        self.get_logger().info(f"/map received: size {mwidth}x{mheight}, origin ({mox:.6f},{moy:.6f}), resolution {mres}")
        self.get_logger().info(f"/map world bounds x[{x_min:.3f},{x_max:.3f}] y[{y_min:.3f},{y_max:.3f}]")


    def pose_callback(self, msg):
        
        self.robot_pose = msg.pose.pose  # NUOVO: salva la pose completa
        
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # --- Covarianze principali AMCL ---
        self.cov_xx = msg.pose.covariance[0]
        self.cov_xy = msg.pose.covariance[1]
        self.cov_yx = msg.pose.covariance[6]
        self.cov_yy = msg.pose.covariance[7]
        self.cov_theta = msg.pose.covariance[35]  # yaw (rotazione attorno a Z)
        if self.map_data is not None:
            resolution = self.map_data.info.resolution
            origin = self.map_data.info.origin
            col = int((x - origin.position.x) / resolution)
            row = int((y - origin.position.y) / resolution)
            self.robot_position = (row, col)

    # stop del robot cancellando il goal corrente
    def stop_robot(self):
        """Ferma IMMEDIATAMENTE il robot cancellando goal Nav2 e pubblicando Twist(0,0,0)"""
        self.get_logger().info("=== STOP ROBOT CHIAMATO ===")
        
        # 1. CANCELLA IL GOAL NAV2 se esiste
        if self.navigating and getattr(self, 'goal_handle', None) is not None:
            try:
                self.get_logger().info("Cancellazione goal Nav2 in corso...")
                cancel_future = self.goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(self.cancel_done_callback)
            except Exception as e:
                self.get_logger().warn(f"Failed to cancel goal: {e}")
        
        # 2. RESETTA TUTTI I FLAG DI NAVIGAZIONE
        self.navigating = False
        self.goal_handle = None
        
        # 3. FERMA I MOTORI IMMEDIATAMENTE (pubblica più volte per sicurezza)
        self.get_logger().info("Pubblicazione comandi STOP ai motori...")
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.linear.y = 0.0
        stop_msg.angular.z = 0.0
        
        # Pubblica 3 volte per essere sicuri
        for _ in range(3):
            self.cmd_vel_pub.publish(stop_msg)
        
        self.get_logger().info("=== ROBOT FERMATO ===")

    def cancel_done_callback(self, future):
        try:
            result = future.result()
            # result may have an enum or return_code attribute depending on implementation
            if hasattr(result, 'return_code') and result.return_code == 2:  # CANCELLED
                self.get_logger().info("Goal cancellato con successo!")
            else:
                self.get_logger().info("Goal cancellation response received")
        except Exception as e:
            self.get_logger().warn(f"cancel_done_callback error: {e}")

    
    def handle_activate_robot(self, request, response):
        # l'attivazione vera e propria viene gestita dal BT
        if not self.is_robot_active:
            # Se la missione è terminata, NON permettere la riattivazione
            if self.mission_ended:
                response.success = False
                response.message = "Cannot reactivate - mission has ended."
                self.get_logger().warn("Activation denied: mission has ended.")
                return response
            
            self.external_activation = True  # Imposta il flag PRIMA di attivare
            response.success = True
            response.message = "Robot activation requested."
            self.get_logger().info("Robot ACTIVATION REQUESTED")
            
            if self.timer.is_canceled():
                self.timer.reset()
            
            # *** RIMOSSO IL CODICE CHE IMPOSTAVA current_goal QUI ***
            # Il ripristino del goal è gestito dal nodo Activation nel BT

        else:
            response.success = True
            response.message = "Robot is already active."
            self.get_logger().warn("Activation called, but robot is already active.")
        return response
    

    def handle_deactivate_robot(self, request, response):
    # La disattivazione vera e propria avviene dentro al BT

        if self.is_robot_active:
            self.get_logger().info("DISATTIVAZIONE IMMEDIATA RICHIESTA!")
            # Imposta i flag per il BT
            self.external_deactivation = True
            
            # FERMA TUTTO IMMEDIATAMENTE
            #self.timer.cancel()
            self.stop_robot()
            
            # SALVA il goal corrente PRIMA di cancellarlo (per eventuale riattivazione)
            if not self.missionComplete and self.current_goal is not None:
                self.saved_goal_before_deactivation = self.current_goal
                self.get_logger().info(f"Salvato goal corrente: {self.current_goal}")
            
            # CANCELLA TUTTI I GOAL E LO STATO DI NAVIGAZIONE
            self.current_goal = None
            self.arrived_goal = None
            self.navigating = False
            self.arrivedGoalNav2 = False
            self.navigationSafe = False
            
            response.success = True
            response.message = "Robot stopped and deactivation requested."
            self.get_logger().info("Robot FERMATO e DEACTIVATION REQUESTED.")
        else:
            response.success = True
            response.message = "Robot is already inactive."
            self.get_logger().warn("Deactivation called, but robot is already inactive.")
        
        return response
    
    def _on_send_goal_response(self, future):
        # callback for send_goal_async; store goal_handle and attach result callback
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn("Nav2: goal rejected by action server")
                self.current_goal = None
                self.goal_handle = None
                self.navigating = False
                return
            self.get_logger().info("Nav2: goal accepted by action server")
            self.goal_handle = goal_handle
            # attach result callback
            goal_handle.get_result_async().add_done_callback(self.navigation_complete_callback)
        except Exception as e:
            self.get_logger().error(f"Error sending goal: {e}")
            self.goal_handle = None
            self.navigating = False

    def navigate_to(self, x, y):
        # IMPORTANTE: Non inviare se già navigando!
        if self.navigating:
            self.get_logger().warn(f"Already navigating - ignoring request to ({x:.2f}, {y:.2f})")
            return True
        
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg() 

        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation.w = 1.0

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_pose

        self.get_logger().info(f'Sending goal to Nav2: ({x:.2f}, {y:.2f})')
        
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Navigate action server not available")
            return False

        self.navigating = True
        send_goal_future = self.nav_client.send_goal_async(nav_goal)
        send_goal_future.add_done_callback(self._on_send_goal_response)
        return True

    def navigation_complete_callback(self, future):
        # future is the get_result_async future
        try:
            goal_result = future.result()
            status = goal_result.status
            # goal_result.result is the result message
            if status == 4:  # SUCCEEDED
                self.get_logger().info("Nav2: goal raggiunto!")
                self.arrivedGoalNav2 = True
                self.navigating = False
            else:
                self.get_logger().warn(f"Nav2: goal non raggiunto, status {status}")
                self.arrivedGoalNav2 = False
        except Exception as e:
            self.get_logger().error(f"Error in navigation_complete_callback: {e}")
            self.arrivedGoalNav2 = False

    def creep_to_goal(self, goal_x, goal_y, tolerance=0.15):
        # Muove lentamente il robot verso il goal finché non è entro la soglia desiderata (default 0.05m).
        # Restituisce True se il robot è arrivato, False altrimenti.
        if self.robot_pose is None:
            self.get_logger().warn("Robot pose not available for creep_to_goal!")
            return False
        
        # Posizione corrente del robot
        x = self.robot_pose.position.x
        y = self.robot_pose.position.y
        
        # Distanza al goal
        dist = math.hypot(goal_x - x, goal_y - y)
        
        self.get_logger().info(f"Creep: distanza={dist:.3f}m (tolleranza={tolerance}m)")
        
        # Se siamo abbastanza vicini, FERMA e ritorna successo
        if dist < tolerance:
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(stop_msg)
            self.get_logger().info("Creep completato - goal raggiunto!")
            return True
        
        # Calcola l'angolo verso il goal (in coordinate mondo)
        angle_to_goal = math.atan2(goal_y - y, goal_x - x)
        
        # Estrai lo yaw corrente dal quaternion
        orientation = self.robot_pose.orientation
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Calcola la differenza angolare (normalizzata tra -π e π)
        angle_diff = angle_to_goal - current_yaw
        while angle_diff > math.pi:
            angle_diff -= 2.0 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2.0 * math.pi
        
        twist = Twist()
        
        # Soglia per decidere se ruotare o muoversi
        ROTATION_THRESHOLD = 0.3  # ~17 gradi
        
        if abs(angle_diff) > ROTATION_THRESHOLD:
            # Il robot NON è orientato verso il goal → RUOTA SUL POSTO
            self.get_logger().info(f"Creep: ruoto di {math.degrees(angle_diff):.1f}°")
            twist.linear.x = 0.0
            twist.angular.z = 0.5 if angle_diff > 0 else -0.5  # Velocità angolare moderata
        else:
            # Il robot è orientato verso il goal → MUOVITI IN AVANTI
            # Velocità proporzionale alla distanza (più lento vicino al goal)
            linear_speed = min(0.1, dist * 0.3)  # Max 0.1 m/s
            angular_correction = angle_diff * 1.0  # Piccole correzioni durante il movimento
            
            self.get_logger().info(f"Creep: avanzo a {linear_speed:.3f}m/s")
            twist.linear.x = linear_speed
            twist.angular.z = angular_correction
        
        self.cmd_vel_pub.publish(twist)
        return False


    def is_robot_lost(self):
            """Controlla se il robot si è perso verificando la covarianza AMCL"""
            # Se non abbiamo dati di covarianza, assumiamo che sia ok
            if (self.cov_xx is None or self.cov_xy is None or 
                self.cov_yx is None or self.cov_yy is None or 
                self.cov_theta is None):
                return False
            
            try:
                pos_cov = np.array([
                    [self.cov_xx, self.cov_xy], 
                    [self.cov_yx, self.cov_yy]
                ])
                
                max_eig = np.max(np.linalg.eigvals(pos_cov))
                yaw_var = self.cov_theta
                
                # Stesse soglie di CheckRobotLost
                if max_eig > 1.0 or yaw_var > 0.5:
                    self.get_logger().warn(
                        f" Covarianza alta! max_eig={max_eig:.3f}, yaw_var={yaw_var:.3f}"
                    )
                    return True
                
                return False
            except Exception as e:
                self.get_logger().error(f"Errore check covarianza: {e}")
                return False


def main(args=None):
    rclpy.init(args=args)
    
    try:
        bt_node = BTStudentsNode()
        rclpy.spin(bt_node)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        if 'bt_node' in locals():
            bt_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


""" per eseguire il tutto:
    # Terminale 1: Simulator
   ros2 launch irob_assignment_5 simulator.launch.py
   
   # Terminale 2: Localization
   ros2 launch nav2_bringup localization_launch.py \
     map:=/home/m/a/mazzani/ros_ws/src/assign_5_IROB_pk/src/irob_assignment_5/maps_saved/my_map.yaml \
     use_sim_time:=true \
     autostart:=true
   
   # Terminale 3: Pubblica initial pose
   ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: -1.0, y: -0.5, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.068]}}"
   
   # Terminale 4: Navigation (aspetta che AMCL pubblichi su /amcl_pose)
   ros2 launch irob_assignment_5 start_navigation.launch.py \
     use_sim_time:=true \
     autostart:=true
   
   # Terminale 5: Il tuo nodo
   ros2 launch irob_assignment_5 project_launch.launch.py
    
 """


""" COMANDI PER ATTIVAZIONE DISATTIVAZIONE E NEW GOAL 
attivazione
ros2 service call /activate irob_interfaces/srv/Activate "{}"

disattivazione
ros2 service call /deactivate irob_interfaces/srv/Deactivate "{}"

getnew goal
ros2 service call /get_goal_external irob_interfaces/srv/GetGoal "{}"
"""
