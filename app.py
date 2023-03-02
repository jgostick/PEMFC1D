import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spop
from scipy import constants as c


state = st.session_state
if not hasattr(state, 'CO2cat'):
    state.CO2cat = 1.0


tabs = st.tabs(["Channel", "GDL", "Membrane", "Catalyst"])

with tabs[0]:
    cols1 = st.columns(2)
    cols1[0].markdown("### Input data")
    Tcat = cols1[0].number_input(
        label='Temperture [K]',
        key='T_cathode',
        value=350,
        step=5,
        min_value=275,
        max_value=372,
    )
    Pcat = cols1[0].number_input(
        label='Pressure [kPa]',
        key='P_cathode',
        value=150,
        step=10,
        min_value=100,
        max_value=500,
    )*1000  # Convert to Pa
    RHcat = cols1[0].number_input(
        label='Relative Humidity [%]',
        key='RH_cathode',
        value=80,
        step=5,
        min_value=0,
        max_value=100,
    )
    cols1[1].markdown("### Conditions")
    A = 8.07131
    B = 1730.63
    C = 233.426
    PV = 10**(A-B/(C+Tcat-273))/760*101325
    cols1[1].text_input(
        label='Vapor Pressure [Pa]',
        value=str(np.around(PV, decimals=0)),
        disabled=True,
    )
    Cgas = Pcat/(c.R*Tcat)
    cols1[1].text_input(
        label='Molar Density [mol/m3]',
        value=str(np.around(Cgas, decimals=0)),
        disabled=True,
    )
    xO2cat = 0.21*(1 - RHcat/100*PV/(Pcat))
    cols1[1].text_input(
        label='Oxygen mol fraction',
        value=str(np.around(xO2cat, decimals=4)),
        disabled=True,
    )
    CO2cat = Cgas*xO2cat
    cols1[1].text_input(
        label='Oxygen concentration [mol/m3]',
        value=str(np.around(CO2cat, decimals=0)),
        disabled=True,
    )


with tabs[1]:
    cols1 = st.columns(2)
    L_GDL = cols1[0].number_input(
        label='GDL Thickness [um]',
        value=300,
        step=5,
        min_value=50,
        max_value=500,
    )*1e-6  # Convert to m
    e = cols1[0].number_input(
        label='GDL Porosity [%]',
        value=75,
        step=5,
        min_value=10,
        max_value=90,
    )/100  # Convert to fraction
    a = cols1[0].slider(
        label='Bruggeman Exponent',
        value=2.0,
        step=0.1,
        min_value=0.0,
        max_value=10.0,
    )
    taux = (e)**-a
    cols1[1].text_input(
        label='Tortuosity',
        value=str(np.around(taux, decimals=3)),
        disabled=True,
    )
    DAB = 0.0000209/(298.0**1.5)*(Tcat**1.5)
    cols1[1].text_input(
        label='Gas Diffusivity [m2/s]',
        value=str(np.around(DAB, decimals=8)),
        disabled=True,
    )
    ilim = 4*96487*e/taux*DAB/L_GDL*(CO2cat-0)/10000
    cols1[1].text_input(
        label='Limiting current [A/cm2]',
        value=str(np.around(ilim, decimals=3)),
        disabled=True,
    )

with tabs[2]:
    cols1 = st.columns(2)
    L_PEM = cols1[0].number_input(
        label='PEM Thickness [um]',
        value=50,
        step=5,
        min_value=10,
        max_value=500,
    )*1e-6
    sigma = cols1[0].slider(
        label='Ionic Conductivity',
        value=10.0,
        step=0.1,
        min_value=0.0010,
        max_value=100.0,
    )

with tabs[3]:
    cols1 = st.columns(2)
    io = cols1[0].number_input(
        label='Exchange Current Density [A/m2]',
        value=1e-3,
        step=0.1,
        min_value=1.0e-5,
        max_value=1.0e5,
        format="%e",
    )
    alpha = cols1[0].slider(
        label='Transfer coefficient',
        value=0.5,
        step=0.05,
        min_value=0.0,
        max_value=1.0,
    )
    z = cols1[1].text_input(
        label='Electrons per mol',
        value=str(4),
        disabled=True,
    )

st.markdown("---")
tabs = st.tabs(["Polarization Curve", "Voltage Profiles"])


# %%
def find_eta(E_cell, eta=None):
    if eta is None:
        eta = E_cell - 1.22
    # Find slope of error at current guess
    eta_calc = []
    eta_guess = []
    eta_guess.append(eta*0.99)
    eta_calc.append(calc_eta(E_cell=E_cell, eta_guess=eta_guess[0]))
    eta_guess.append(eta*1.01)
    eta_calc.append(calc_eta(E_cell=E_cell, eta_guess=eta_guess[0]))

    for i in range(1, 10):
        # Use Newton's method to guess next eta
        err_1 = eta_guess[i-1] - eta_calc[i-1]
        err_2 = eta_guess[i] - eta_calc[i]
        m = (err_2 - err_1)/(eta_guess[i] - eta_guess[i-1])
        eta_guess.append((0 - err_2)/m + eta_guess[i])
        eta_calc.append(calc_eta(E_cell=E_cell, eta_guess=eta_guess[i]))
        # if m**2 < 1e-10:
            # break
    return eta_calc[-1]


def eta_error(eta, E_cell):
    eta_calc = calc_eta(E_cell=E_cell, eta_guess=eta)
    error = (eta_calc - eta)**2
    return error


def calc_eta(E_cell, eta_guess):
    i = calc_i(E_cell=E_cell, eta=eta_guess)
    phi_e = E_cell
    phi_H_calc = 0 - i*(L_PEM)/sigma
    eta_calc = phi_e - phi_H_calc - 1.22
    return eta_calc


def calc_i(E_cell, eta):
    k = io*np.exp(-4*96487*(1-alpha)*eta/(8.314*Tcat))
    g = 4*96487*DAB*(e/taux)/(L_GDL)
    CO2_CL = g*CO2cat/(g + k)
    i = k*CO2_CL
    # i = g*(CO2cat - CO2_CL)
    return i


E_cell = np.arange(1.2, 0.0, -0.05)
i_cell = []
for E in E_cell:
    eta = spop.fmin(func=eta_error, x0=-0.1, args=(E, ))
    i = calc_i(E_cell=E, eta=eta)
    i_cell.append(i)

fig, ax = plt.subplots()
ax.plot(i_cell, E_cell, 'b-o')
tabs[0].pyplot(fig)

with tabs[1]:
    E = st.slider(
        label='Cell Voltage',
        value=1.0,
        step=0.05,
        min_value=0.1,
        max_value=1.2,
    )


