import streamlit as st
import openpnm as op
import matplotlib.pyplot as plt
from copy import copy
from scipy import constants as c


state = st.session_state
if not hasattr(state, 'CO2cat'):
    state.CO2cat = 1.0

tabs = st.tabs(["Conditions", "Dimensions"])

with tabs[0]:
    cols0 = st.columns(2)
    cols0[0].markdown("**Cathode**")
    Tcat = cols0[0].number_input(
        label='Temperture [K]',
        key='T_cathode',
        value=350,
        step=5,
        min_value=275,
        max_value=372,
    )
    Pcat = cols0[0].number_input(
        label='Pressure [kPa]',
        key='P_cathode',
        value=150,
        step=10,
        min_value=100,
        max_value=500,
    )
    RHcat = cols0[0].number_input(
        label='Relative Humidity [%]',
        key='RH_cathode',
        value=80,
        step=5,
        min_value=0,
        max_value=100,
    )
    xO2cat = 0.21*(1 - RHcat/100)
    Ccat = 101.323*Pcat/(c.R*Tcat)
    CO2cat = Ccat*xO2cat

    cols0[1].markdown("**Anode**")
    Tan = cols0[1].number_input(
        label='Temperture [K]',
        key='T_anode',
        value=Tcat,
        step=5,
        min_value=275,
        max_value=372,
        disabled=True,
    )
    Pan = cols0[1].number_input(
        label='Pressure [kPa]',
        key='P_anode',
        value=150,
        step=10,
        min_value=101,
        max_value=300,
    )
    RHan = cols0[1].number_input(
        label='Relative Humidity [%]',
        key='RH_anode',
        value=80,
        step=5,
        min_value=0,
        max_value=100,
    )

with tabs[1]:
    cols1 = st.columns(2)
    L = cols1[0].number_input(
        label='GDL Thickness [um]',
        value=300,
        step=5,
        min_value=50,
        max_value=500,
    )
    e = cols1[0].number_input(
        label='GDL Porosity [%]',
        value=75,
        step=5,
        min_value=10,
        max_value=90,
    )
    A = cols1[0].number_input(
        label='Anisotropy Ratio [%]',
        help='The ratio of in-plane to through-plane transport rate',
        value=1.0,
        step=0.1,
        min_value=0.01,
        max_value=10.0,
    )
    taux = (e/100)**-2
    tauy = taux/A
    cols1[0].text_input(
        label='Tortuosity',
        value=str(taux) + ' | ' + str(tauy),
        disabled=True,
    )

# %%
state.shape = [20, 100]
if st.button(label='Run Simulation'):
    pn = op.network.Cubic(state.shape, spacing=1)
    pn.add_model(propname='throat.vector',
                 model=op.models.geometry.throat_vector.pore_to_pore)
    air = op.phase.Air(network=pn)
    air['pore.temperature'] = Tcat
    air['pore.pressure'] = Pcat
    air.regenerate_models()
    Tsx = pn['throat.vector'][:, 0] == 1
    air['throat.diffusive_conductance'] = 0.0
    air['throat.diffusive_conductance'][Tsx] = air['throat.diffusivity'][Tsx]*e/taux
    Tsy = pn['throat.vector'][:, 1] == 1
    air['throat.diffusive_conductance'][Tsy] = air['throat.diffusivity'][Tsy]*e/tauy


    m = op.models.physics.source_terms.standard_kinetics
    air.add_model(propname='pore.reaction',
                  model=m,
                  X='pore.concentration',
                  prefactor=-0.01,
                  exponent=1,
                  regen_mode='deferred')

    fd = op.algorithms.FickianDiffusion(network=pn, phase=air)
    Ps = pn['pore.right']*(pn.coords[:, 1] < 80)*(pn.coords[:, 1] > 20)
    fd.set_value_BC(pores=Ps, values=CO2cat)
    fd.set_source(pores=pn['pore.left'], propname='pore.reaction')
    fd.run()
    state.conc = fd['pore.concentration']
    state.CO2cat = CO2cat



if hasattr(state, 'conc'):
    fig, ax = plt.subplots()
    ax.axis(False)
    im = ax.imshow(state.conc.reshape(state.shape),
                   interpolation='bilinear',
                   cmap=plt.cm.plasma,
                   vmin=0, vmax=state.CO2cat)
    fig.colorbar(im)
    st.pyplot(fig)



