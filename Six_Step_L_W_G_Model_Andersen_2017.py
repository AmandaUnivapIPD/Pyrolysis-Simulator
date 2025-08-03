
"""
Created on Jun 23, 2024

@author: Amanda Arthuzo Corrêa

Modelo baseado em Arndersen, 2017 

"""
#%%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%%

# Constantes do Polimero para valores individuais, obtidas da 'Table 4-1 - Kinetic parameters for decomposition of polymers, pg 34' 
# Para o cálculo das frações, os dados foram obtidos de 'Table 4 - Kinetic parameters for some reaction pathways. Ding 2012

'''
ln(𝑘) = ln(𝐴0) −𝐸𝑎/𝑅 * (1/𝑇) 

Thus the A0 can be calculated taking the exponential of where the line crosses the y-axis, and 
the slope is the activation energy (Ea) divided by the gas constant.

import numpy as np

# Se x = 1/T e y = ln(k):
x = np.array([x1, x2, x3, x4])  
y = np.array([y1, y2, y3, y4])  
# Calcule a média de x e y
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calcule as diferenças
x_diff = x - x_mean
y_diff = y - y_mean

# Calcule o coeficiente angular (m)
m = np.sum(x_diff * y_diff) / np.sum(x_diff**2)

# Calcule o coeficiente linear (b)
b = y_mean - m * x_mean

print(f"O coeficiente angular é {m} e o coeficiente linear é {b}")


Table 3-1 - Rate constants HDPE (min-1) - Andersen2017

T;    k1;      k2;       k3;       k4;       k5
360;  0.0034;  0.0005;   0.0001;   0.0003;   0.0016
380;  0.01;    0.0016;   0.001;    0.0002;   0.0003
400;  0.0338;  0.0006;   0.002;    0.002     0.0041
420;  0.1248;  0.0131;   0.0089;   0.0147;   0.0094


T_k = [360,380,400,420]
k_xp = [[0.0034,0.01,0.0338,0.1248],
        [0.0005,0.0016,0.0006,0.0131],
        [0.0001,0.001,0.002,0.0089],
        [0.0003,0.0002,0.002,0.0147],
        [0.0016,0.0003,0.0041,0.0094]]

'''
R = 8.314  # Constante universal dos gases, J/(mol K)
A0_k = [4.2E16,3.3E13,1.3E18,2.36E11,1.4572443,1.32E-64]
Ea_R = [-25810.6859,-22181.9896,-29441.7479,-19301.9116,-4230.9154,74873.1962]
Ea_k = x = [abs(i*R) for i in Ea_R]



# Constantes do Polimero, obtidas da 'Table 4-1 - Kinetic parameters for decomposition of polymers, pg 34' 
# Ea_poly = 216.15E3 # J/mol
# A0_poly = 2.53E15

Ea_poly = 158.62E3 # J/mol
A0_poly = 1.23E11   

# Ea_PP = 158.62E3 # J/mol
# A0_PP = 1.23E11   

# Ea_HDPE = 216.15E3 # J/mol
# A0_HDPE = 2.53E15

# Ea_MIX = 219.58E3 # J/mol
# A0_MIX = 1.56E17

#%%
# Definindo o sistema de EDOs incluindo
# Os parametros iniciais estão representados por percentual %
# Na  versão final, realizar as conversões para massa e mol
# As temperaturas estão convertidas para Kelvin, mas serão exibidas em °C

# Condições iniciais
Per_XP = 100 # Percentual inicial de Polimero
Per_XL = 0
Per_XW = 0
Per_XG = 0


initial_temperature_celcius = 300
initial_temperature = initial_temperature_celcius + 273  # Temperatura inicial (K)
final_temperature_celcius = 400
final_temperature =  final_temperature_celcius + 273 #Temperatura final (K)
taxa_de_aquecimento_celcius = 10 # ºC/min
taxa_aquecimento = 50 # ºC/min
tempo_de_aquecimento = final_temperature_celcius/taxa_aquecimento # Tempo em minutos no qual o reator sai de initial_temperature para final_temperature em com taxa_de_aquecimento_celcius

tempo_de_residencia = 0 #a ser considerado posteriormente

#%%
def arrhenius(A,Ea,R,T):
    return A*np.exp(-Ea/(R*T))
#%%
def ajuste_de_linha(kelvin, y):
    # Calculate slope and intercept
    
    x = [1/i for i in kelvin]
    y = np.log(y)

    # Calcular os coeficientes da reta usando o método dos mínimos quadrados
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return m, b
#%%
# Ea_k = []
# A0_k = []
# T_Kelvin = [273+i for i in T_k]
# for k in k_xp:
#     m, b = ajuste_de_linha(T_Kelvin, k)
#     Ea_k.append(abs(m*R))
#     A0_k.append(np.exp(b))

# print("Ea_k:", Ea_k)
# print("A0_k:", A0_k)

#%%
def pyrolysis_model(y, t):
    XP, XL, XW, XG, delta_temperatura = y

    k1 = arrhenius(A0_k[0],Ea_k[0],R, delta_temperatura )
    k2 = arrhenius(A0_k[1],Ea_k[1],R, delta_temperatura )
    k3 = arrhenius(A0_k[2],Ea_k[2],R, delta_temperatura )
    k4 = arrhenius(A0_k[3],Ea_k[3],R, delta_temperatura )
    k5 = arrhenius(A0_k[4],Ea_k[4],R, delta_temperatura )
    k6 = arrhenius(A0_k[5],Ea_k[5],R, delta_temperatura )
    # Taxa de variação da temperatura (exemplo de aquecimento linear)
    dT_dt = taxa_aquecimento  
    
    # Taxas de variação das concentrações
    dXP_dt = - XP * (k1 + k2 + k3)
    dXL_dt = XP*k2 + XW*k4 - XL*k6
    dXW_dt = XP*k1 - XW*(k5 + k4) # Acho que é k4 no lugar de k6 ....
    dXG_dt = XP*k3 + XL*k6 + XW*k5


    
    
    return [dXP_dt,dXL_dt,dXW_dt,dXG_dt, dT_dt]

#%%

#%%
# Condições iniciais para o sistema de EDOs
y0 = [Per_XP, Per_XL, Per_XW, Per_XG, initial_temperature]

# Intervalo de tempo para a simulação
t = np.linspace(0, tempo_de_aquecimento,50)  # Tempo de residência divididos em 500 pontos

#%%
# Resolvendo as EDOs
solution = odeint(pyrolysis_model, y0, t)
poly_concentration = solution[:, 0]
L_Concentration = solution[:, 1]
W_Concentration = solution[:, 2]
G_Concentration = solution[:, 3]


temperature = solution[:, 4]

#%%
# Plotando os resultados
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(temperature-273, poly_concentration, label='Polímero Artigo', color='blue',linestyle='--')
plt.plot(temperature-273, poly_concentration, label='Polímero Simulador', color='blue', marker='s', linestyle='None')
plt.plot(temperature-273, L_Concentration, label='L Artigo', color='green',linestyle='--')
plt.plot(temperature-273, L_Concentration, label='L Simulador', color='green',marker='*',linestyle='None')
plt.plot(temperature-273, W_Concentration, label='W Simulador', color='orange',linestyle='--')
plt.plot(temperature-273, W_Concentration, label='W Artigo', color='orange',marker='o',linestyle='None')
plt.plot(temperature-273, G_Concentration, label='G Simulador', color='red',linestyle='--')
plt.plot(temperature-273, G_Concentration, label='G Artigo', color='red',marker='D',linestyle='None')
plt.ylabel('Percentual de Polímero, L, W, G (%)')
plt.title('Simulação da Pirólise - 6 Lump model ')
plt.legend()
plt.axis('tight')
plt.grid(True)

'''
plt.subplot(2, 1, 2)
plt.plot(t, temperature - 273, label='Temperatura', color='purple')
plt.xlabel('Tempo (min)')
plt.ylabel('Temperatura (°C)')
plt.title('Variação da Temperatura durante a Pirólise')
plt.legend()
plt.grid(True)
plt.tight_layout()
'''
plt.show()

# %%
