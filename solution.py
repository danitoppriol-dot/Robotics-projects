#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# {daniele priola}
# {priola}
# {priola@kth.se}

from dubins import step
import math

def ipotenusa(x1, x2, y1, y2):
    dx = x1 - x2
    dy = y1 - y2
    return (dx*dx + dy*dy) ** 0.5

def prendi_f(n):
    return n[0]

def solution(car):
    dt = 0.01
    passi = 100         
    raggio_obb = 1.5   
    margine = 0.25 
    h0 = ipotenusa(car.x0, car.xt, car.y0, car.yt)
    open_list = [(h0, 0.0, car.x0, car.y0, 0.0, [], [0.0])]
    visited = {}

    while len(open_list) > 0:
        open_list.sort(key=prendi_f)
        f, g, x, y, theta, controls, times = open_list.pop(0)
        if ipotenusa(x, car.xt, y, car.yt) <= raggio_obb:
            return controls, times
        for phi in [-math.pi/4, 0.0, math.pi/4]:
            cx = x
            cy = y
            ctheta = theta
            nuovi_controls = controls.copy()
            nuovi_times = times.copy()
            costo_aggiunto = 0.0
            valido = True
            for i in range(passi):
                cx, cy, ctheta = step(car, cx, cy, ctheta, phi, dt)
                
                if ctheta > math.pi:
                    ctheta = ctheta - 2*math.pi
                elif ctheta < -math.pi:
                    ctheta = ctheta + 2*math.pi
                nuovi_controls.append(phi)
                nuovi_times.append(nuovi_times[-1] + dt)
                costo_aggiunto += dt

                if not (car.xlb <= cx <= car.xub and car.ylb <= cy <= car.yub):
                    valido = False
                    break
                for (ox, oy, r) in car.obs:
                    if ipotenusa(cx, ox, cy, oy) <= r + margine:
                        valido = False
                        break
                if ipotenusa(cx, car.xt, cy, car.yt) <= raggio_obb:
                    return nuovi_controls, nuovi_times
                
            if valido:
                h = ipotenusa(cx, car.xt, cy, car.yt)
                nuovo_g = g + costo_aggiunto
                chiave = (round(cx,0), round(cy,0), round(ctheta,0))
                if chiave not in visited or visited[chiave] > nuovo_g:
                    visited[chiave] = nuovo_g
                    open_list.append((nuovo_g+h, nuovo_g, cx, cy, ctheta, nuovi_controls, nuovi_times))
                    
    return [0.0], [0.0, dt]
