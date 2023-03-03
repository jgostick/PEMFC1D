import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spop
from scipy import constants as c


state = st.session_state
if not hasattr(state, 'CO2cat'):
    state.CO2cat = 1.0

st.title("PEMFC Simulator :zap:")
st.subheader(":chart_with_downwards_trend: First Principles, 1D Model")

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
    with st.expander('Transport through the GDL is governed by Fick\'s law:'):
        st.latex(r'n_{O_2} = \frac{\varepsilon}{\tau} \frac{D_{AB}}{L_{GDL}} (C_{O_2,CL} - C_{O_2,Ch})')
        st.markdown('where:')
        st.markdown(r" - $$ \varepsilon $$ is the porosity")
        st.markdown(r" - $$ \tau $$ is the tortuosity")
        st.markdown(r" - $$ L_{GDL} $$ is the thickness of the GDL [m]")
        st.markdown(r" - $$ D_{AB} $$ is the diffusion coefficient of $O_2$ in air [$m^2/s$]")
        st.markdown(r" - $$ C_{O_2} $$ is the concentration of oxygen [$mol/m^3$]")
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
    with st.expander("Transport through the membrane is governed by Ohm's law:"):
        st.latex(r'i = \frac{\sigma}{L_{PEM}} (\phi_{H^+,An} - \phi_{H^+,Cat})')
        st.markdown('where:')
        st.markdown(r" - $$ \sigma $$ is the ionic conductivity [S/m]")
        st.markdown(r" - $$ L_{PEM} $$ is the thickness of the membrane [m]")
        st.markdown(r" - $$ \phi_{H^+} $$ is the potential in the ionmer [V]")
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
        value=5.0,
        step=0.1,
        min_value=0.0010,
        max_value=20.0,
    )

with tabs[3]:
    with st.expander("Electrochemical kinetics are described by the Tafel equation"):
        st.latex(r'i = i_o \cdot C_{A,CL} exp \bigg[-\frac{(1 - \alpha)zF}{RT} \eta \bigg]')
        st.markdown(r" - $$ \alpha $$ is the Transfer Coefficient")
        st.markdown(r" - $$ i_o $$ is the Exhange Current Density")
        st.markdown(r" - $$ z $$ is the number of electrons per molecule")
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


# %%
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

st.markdown("---")
tabs = st.tabs(["Polarization Curve", "Voltage Profile", "Oxygen Profile"])

with tabs[0]:
    fig, ax = plt.subplots()
    ax.plot(np.array(i_cell, dtype=float), np.array(E_cell, dtype=float), 'b-o')
    ax.set_xlabel('Current Density [$A/m^2$]')
    ax.set_ylabel('Cell Voltage [$V$]')
    st.pyplot(fig)

with tabs[1]:
    E = st.slider(
        label='Cell Voltage',
        value=1.0,
        step=0.05,
        min_value=0.0,
        max_value=1.22,
    )
    eta = spop.fmin(func=eta_error, x0=-0.1, args=(E, ))
    i = calc_i(E_cell=E, eta=eta)
    phi_e = E
    phi_H_calc = 0 - i*(L_PEM)/sigma

    # Plot voltage profile
    fig, ax = plt.subplots()
    ax.plot(np.array([0.0, L_PEM*1e6], dtype=float), np.array([0.0, phi_H_calc], dtype=float), 'b-o')
    ax.plot(np.array([L_PEM*1e6, L_PEM*1e6], dtype=float), np.array([phi_H_calc, phi_e], dtype=float), 'r-o')
    ax.plot(np.array([L_PEM*1e6, L_PEM*1e6 + L_GDL*1e6], dtype=float), np.array([phi_e, phi_e], dtype=float), 'g-o')
    ax.plot(np.array([0.0, L_PEM*1e6 + L_GDL*1e6], dtype=float), np.array([1.22, 1.22], dtype=float), 'm--o')
    ax.set_xlabel('Distance From Anode CL [$\mu m$]')
    ax.set_ylabel('Cell Voltage [$V$]')
    ax.set_ylim([-1.0, 1.5])
    st.pyplot(fig)

with tabs[2]:
    E = st.slider(
        label='Cell Voltage',
        key='Voltage for O2 tab',
        value=1.0,
        step=0.05,
        min_value=0.0,
        max_value=1.22,
    )
    eta = spop.fmin(func=eta_error, x0=-0.1, args=(E, ))
    k = io*np.exp(-4*96487*(1-alpha)*eta/(8.314*Tcat))
    g = 4*96487*DAB*(e/taux)/(L_GDL)
    CO2_CL = g*CO2cat/(g + k)

    # Plot voltage profile
    fig, ax = plt.subplots()
    ax.plot(np.array([0.0, L_PEM*1e6], dtype=float), np.array([0.0, 0.0], dtype=float), 'b-o')
    ax.plot(np.array([L_PEM*1e6, L_PEM*1e6 + L_GDL*1e6], dtype=float), np.array([CO2_CL, CO2cat], dtype=float), 'g-o')
    ax.plot(np.array([0.0, L_PEM*1e6 + L_GDL*1e6], dtype=float), np.array([0.21*Cgas, 0.21*Cgas], dtype=float), 'm--o')
    ax.set_xlabel('Distance From Anode CL [$\mu m$]')
    ax.set_ylabel('Oxygen Concentration [$mol/m^3$]')
    # ax.set_ylim([-1.0, 1.5])
    st.pyplot(fig)





























