"""
Integrantes:
    *Caizatoa Fernanda
    * Llumiquinga Fernando
    * Revelo Alava Héctor
    * Salazar Hugo
    * Yepez Gandhy
"""
# -*- coding: utf-8 -*-
"""
Centros notables de un triángulo (G, I, H) con Streamlit + Plotly


Fórmulas:
- Lados: a = |BC|, b = |CA|, c = |AB|
- Centroide: G = ((xA+xB+xC)/3, (yA+yB+yC)/3)
- Incentro:  I = ( (a*xA + b*xB + c*xC)/(a+b+c), (a*yA + b*yB + c*yC)/(a+b+c) )
- Ortocentro: intersección de dos alturas (perpendiculares a lados, pasando por el vértice)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

# ==============================
# Configuración de página
# ==============================
st.set_page_config(page_title="Centros de un Triángulo", layout="wide", initial_sidebar_state="expanded")
st.title("📐 Centros Notables de un Triángulo: Centroide (G), Incentro (I) y Ortocentro (H)")
st.caption("Ingresa las coordenadas de los vértices A, B y C. La app valida el triángulo, calcula los centros y los visualiza a escala.")

# ==============================
# Utilidades geométricas
# ==============================
def distancia(P, Q):
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    return float(np.linalg.norm(P - Q))

def area_triangulo(A, B, C):
    x1, y1 = A; x2, y2 = B; x3, y3 = C
    return 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))  # fórmula del "zapatero"

def pendiente(P, Q):
    x1, y1 = P; x2, y2 = Q
    if np.isclose(x1, x2):
        return None  # vertical
    return (y2 - y1) / (x2 - x1)

def recta_por_pendiente_y_punto(m, P):
    x0, y0 = P
    if m is None:              # x = x0
        return (1.0, 0.0, -x0)
    # y - y0 = m(x - x0) -> m x - y + (y0 - m x0) = 0
    A = m; B = -1.0; C = y0 - m*x0
    return (A, B, C)

def recta_perpendicular_por_punto(recta, P):
    A, B, _ = recta
    if np.isclose(B, 0):       # recta vertical → perpendicular horizontal (m=0)
        m_perp = 0.0
    elif np.isclose(A, 0):     # recta horizontal → perpendicular vertical
        m_perp = None
    else:
        m = -A / B
        m_perp = None if np.isclose(m, 0) else -1.0/m
    return recta_por_pendiente_y_punto(m_perp, P)

def interseccion(r1, r2):
    A1, B1, C1 = r1
    A2, B2, C2 = r2
    D = A1*B2 - A2*B1
    if np.isclose(D, 0):
        return None
    x = (B1*C2 - B2*C1) / D
    y = (C1*A2 - C2*A1) / D
    return (float(x), float(y))

def angulo_interno(a, b, c):
    # Ángulo opuesto a a (ley de cosenos)
    val = (b*b + c*c - a*a) / (2*b*c)
    val = np.clip(val, -1.0, 1.0)
    return np.degrees(np.arccos(val))

# ==============================
# Entrada de datos
# ==============================
st.sidebar.header("Coordenadas de vértices")
colA, colB, colC = st.sidebar.columns(3)
with colA:
    xA = st.number_input("xA", value=0.0, step=0.5, format="%.4f")
    yA = st.number_input("yA", value=0.0, step=0.5, format="%.4f")
with colB:
    xB = st.number_input("xB", value=5.0, step=0.5, format="%.4f")
    yB = st.number_input("yB", value=0.0, step=0.5, format="%.4f")
with colC:
    xC = st.number_input("xC", value=2.0, step=0.5, format="%.4f")
    yC = st.number_input("yC", value=4.0, step=0.5, format="%.4f")

A = (xA, yA); B = (xB, yB); C = (xC, yC)

# Validación: no colineales
S = area_triangulo(A, B, C)
if np.isclose(S, 0.0):
    st.error("⚠️ Los puntos A, B y C son colineales. No definen un triángulo.")
    st.stop()

# Lados y métricas
a = distancia(B, C)   # opuesto a A
b = distancia(C, A)   # opuesto a B
c = distancia(A, B)   # opuesto a C
p = a + b + c
area = S
Aang = angulo_interno(a, b, c)
Bang = angulo_interno(b, c, a)
Cang = angulo_interno(c, a, b)
r_in = area / (0.5*p)  # radio de la circunferencia inscrita

# Centros
G = ((xA + xB + xC)/3.0, (yA + yB + yC)/3.0)
I = ((a*xA + b*xB + c*xC)/p, (a*yA + b*yB + c*yC)/p)

# Ortocentro: intersección de dos alturas
mBC = pendiente(B, C); recta_BC = recta_por_pendiente_y_punto(mBC, B)
mAC = pendiente(A, C); recta_AC = recta_por_pendiente_y_punto(mAC, A)
altura_A = recta_perpendicular_por_punto(recta_BC, A)
altura_B = recta_perpendicular_por_punto(recta_AC, B)
H = interseccion(altura_A, altura_B)
if H is None:
    st.error("No se pudo calcular el ortocentro por paralelismo numérico de las alturas.")
    st.stop()

# ==============================
# KPIs (dos filas: métricas y ángulos)
# ==============================
fila1 = st.columns(3)
fila1[0].metric("Perímetro", f"{p:.4f}")
fila1[1].metric("Área", f"{area:.4f}")
fila1[2].metric("Radio inscrita (r)", f"{r_in:.4f}")

st.markdown("### Ángulos (°)")
fila2 = st.columns(3)
fila2[0].metric("∠A", f"{Aang:.2f}°")
fila2[1].metric("∠B", f"{Bang:.2f}°")
fila2[2].metric("∠C", f"{Cang:.2f}°")

# Tabla compacta de ángulos (apoyo visual)
st.dataframe(
    pd.DataFrame({"Vértice": ["A", "B", "C"],
                  "Ángulo (°)": [round(Aang, 2), round(Bang, 2), round(Cang, 2)]}),
    use_container_width=True
)
# Fórmula en LaTeX
st.latex(r"\angle A = %.2f^\circ,\ \angle B = %.2f^\circ,\ \angle C = %.2f^\circ" % (Aang, Bang, Cang))

st.divider()

# ==============================
# Tabla de resultados (coordenadas de centros)
# ==============================
st.subheader("Resultados numéricos de centros")
df_resultados = pd.DataFrame({
    "Centro": ["Centroide (G)", "Incentro (I)", "Ortocentro (H)"],
    "x": [round(G[0], 6), round(I[0], 6), round(H[0], 6)],
    "y": [round(G[1], 6), round(I[1], 6), round(H[1], 6)]
})
st.dataframe(df_resultados, use_container_width=True)

# ==============================
# Gráfico mejorado a escala
# ==============================
st.subheader("Visualización del triángulo y centros")

xs = [xA, xB, xC, xA]
ys = [yA, yB, yC, yA]

fig = go.Figure()

# Triángulo con rótulos A, B, C
fig.add_trace(go.Scatter(
    x=xs, y=ys, mode="lines+markers+text",
    name="Triángulo ABC",
    text=["A", "B", "C", "A"],
    textposition="top center",
    marker=dict(size=10)
))

# Centros notables
fig.add_trace(go.Scatter(x=[G[0]], y=[G[1]], mode="markers+text",
                         name="G (Centroide)",
                         text=["G"], textposition="bottom right",
                         marker=dict(size=14, symbol="cross")))
fig.add_trace(go.Scatter(x=[I[0]], y=[I[1]], mode="markers+text",
                         name="I (Incentro)",
                         text=["I"], textposition="bottom right",
                         marker=dict(size=14, symbol="diamond")))
fig.add_trace(go.Scatter(x=[H[0]], y=[H[1]], mode="markers+text",
                         name="H (Ortocentro)",
                         text=["H"], textposition="bottom right",
                         marker=dict(size=14, symbol="triangle-up")))

# Alturas dibujadas (opcional para entender H)
def puntos_recta(recta, xs, ys, margen=1.0):
    A1, B1, C1 = recta
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if np.isclose(B1, 0):  # vertical
        x = -C1 / A1
        return [x, x], [ymin-margen, ymax+margen]
    x_vals = np.array([xmin-margen, xmax+margen], dtype=float)
    y_vals = -(A1*x_vals + C1)/B1
    return x_vals.tolist(), y_vals.tolist()

x_altA, y_altA = puntos_recta(altura_A, xs+[G[0], I[0], H[0]], ys+[G[1], I[1], H[1]])
x_altB, y_altB = puntos_recta(altura_B, xs+[G[0], I[0], H[0]], ys+[G[1], I[1], H[1]])
fig.add_trace(go.Scatter(x=x_altA, y=y_altA, mode="lines", name="Altura por A", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=x_altB, y=y_altB, mode="lines", name="Altura por B", line=dict(dash="dash")))

# Rango y proporción 1:1
minx = min(xs + [G[0], I[0], H[0]]); maxx = max(xs + [G[0], I[0], H[0]])
miny = min(ys + [G[1], I[1], H[1]]); maxy = max(ys + [G[1], I[1], H[1]])
margen = 1.0
fig.update_layout(
    xaxis=dict(range=[minx-margen, maxx+margen], scaleanchor="y", scaleratio=1),
    yaxis=dict(range=[miny-margen, maxy+margen]),
    legend=dict(orientation="h", y=1.02, x=1, yanchor="bottom", xanchor="right"),
    margin=dict(l=20, r=20, t=30, b=20),
    width=900, height=680
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# Descarga de resultados
# ==============================
st.download_button(
    "⬇️ Descargar resultados (.txt)",
    data=(
        f"A={A}\nB={B}\nC={C}\n\n"
        f"Perímetro={p:.6f}\nÁrea={area:.6f}\n"
        f"G={G}\nI={I}\nH={H}\n"
        f"Ángulos (°): A={Aang:.4f}, B={Bang:.4f}, C={Cang:.4f}\n"
        f"Radio inscrita r={r_in:.6f}\n"
    ).encode("utf-8"),
    file_name="centros_triangulo.txt",
    mime="text/plain",
)

st.caption("El archivo contiene los datos calculados")

