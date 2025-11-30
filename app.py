import streamlit as st
import json
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Red Eléctrica Lima - MST Kruskal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CLASES Y FUNCIONES ==========

class UFDS:
    """Union-Find Disjoint Set para detección de ciclos"""
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.rank = [0] * (n + 1)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def kruskal_mst(num_nodes, edges):
    """Algoritmo de Kruskal para MST"""
    edges_sorted = sorted(edges, key=lambda x: x['cost'])
    ufds = UFDS(num_nodes)
    mst = []
    total_cost = 0

    for edge in edges_sorted:
        if ufds.union(edge['from'], edge['to']):
            mst.append(edge)
            total_cost += edge['cost']
            if len(mst) == num_nodes - 1:
                break

    return mst, total_cost

def distance(lat1, lon1, lat2, lon2):
    """Distancia Haversine en km"""
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ========== ZONAS DE LIMA ==========
zones = [
    {"name": "Cercado de Lima", "lat": -12.046374, "lon": -77.042793, "poles": 150, "type": "urbano"},
    {"name": "Miraflores", "lat": -12.119470, "lon": -77.028236, "poles": 120, "type": "urbano"},
    {"name": "San Isidro", "lat": -12.095820, "lon": -77.034590, "poles": 100, "type": "urbano"},
    {"name": "Surco", "lat": -12.148240, "lon": -77.012480, "poles": 130, "type": "urbano"},
    {"name": "La Molina", "lat": -12.082770, "lon": -76.942150, "poles": 120, "type": "urbano"},
    {"name": "San Borja", "lat": -12.092890, "lon": -76.996250, "poles": 90, "type": "urbano"},
    {"name": "Surquillo", "lat": -12.113890, "lon": -77.018750, "poles": 80, "type": "urbano"},
    {"name": "Barranco", "lat": -12.146390, "lon": -77.021110, "poles": 70, "type": "urbano"},
    {"name": "Chorrillos", "lat": -12.168750, "lon": -77.012500, "poles": 110, "type": "urbano"},
    {"name": "San Juan de Miraflores", "lat": -12.159720, "lon": -76.973330, "poles": 100, "type": "semi_rural"},
    {"name": "Villa El Salvador", "lat": -12.215830, "lon": -76.939167, "poles": 100, "type": "semi_rural"},
    {"name": "Villa María del Triunfo", "lat": -12.164167, "lon": -76.935833, "poles": 90, "type": "semi_rural"},
    {"name": "Ate", "lat": -12.041111, "lon": -76.905556, "poles": 120, "type": "urbano"},
    {"name": "Santa Anita", "lat": -12.047500, "lon": -76.972778, "poles": 80, "type": "urbano"},
    {"name": "El Agustino", "lat": -12.039444, "lon": -77.012222, "poles": 140, "type": "urbano"}
]

# ========== GENERACIÓN DE DATOS ==========
@st.cache_data
def generar_red_electrica(zones, seed=42):
    """Genera nodos y aristas de la red eléctrica"""
    random.seed(seed)
    
    # Generar nodos
    nodes = []
    node_id = 1
    for zone in zones:
        for i in range(zone["poles"]):
            node = {
                "id": node_id,
                "zone": zone["name"],
                "lat": zone["lat"] + (random.random() - 0.5) * 0.02,
                "lon": zone["lon"] + (random.random() - 0.5) * 0.02,
                "type": zone["type"]
            }
            nodes.append(node)
            node_id += 1

    # Generar aristas
    edges = []
    edge_set = set()

    # Conexiones dentro de cada zona
    for zone in zones:
        zone_nodes = [n for n in nodes if n["zone"] == zone["name"]]
        for node in zone_nodes:
            num_connections = random.randint(3, 6)
            for _ in range(num_connections):
                target = random.choice(zone_nodes)
                if node["id"] != target["id"]:
                    edge_key = tuple(sorted([node["id"], target["id"]]))
                    if edge_key not in edge_set:
                        dist = distance(node["lat"], node["lon"], target["lat"], target["lon"])
                        terrain_factor = 1.5 if zone["type"] == "semi_rural" else 1.0
                        cost = round(dist * 1000 * terrain_factor, 2)
                        edges.append({
                            "from": node["id"],
                            "to": target["id"],
                            "distance": round(dist * 1000, 2),
                            "cost": cost
                        })
                        edge_set.add(edge_key)

    # Conexiones entre zonas
    for i in range(len(zones) - 1):
        zone1_nodes = [n for n in nodes if n["zone"] == zones[i]["name"]]
        zone2_nodes = [n for n in nodes if n["zone"] == zones[i+1]["name"]]
        for _ in range(random.randint(5, 10)):
            n1 = random.choice(zone1_nodes)
            n2 = random.choice(zone2_nodes)
            edge_key = tuple(sorted([n1["id"], n2["id"]]))
            if edge_key not in edge_set:
                dist = distance(n1["lat"], n1["lon"], n2["lat"], n2["lon"])
                cost = round(dist * 1000 * 1.8, 2)
                edges.append({
                    "from": n1["id"],
                    "to": n2["id"],
                    "distance": round(dist * 1000, 2),
                    "cost": cost
                })
                edge_set.add(edge_key)

    return nodes, edges

# ========== VISUALIZACIÓN ==========
def plot_graph(nodes, edges, title, mst_edges=None, selected_zone=None):
    """Genera visualización del grafo"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    G = nx.Graph()
    for node in nodes:
        if selected_zone and node["zone"] != selected_zone:
            continue
        G.add_node(node["id"], pos=(node["lon"], node["lat"]), zone=node["zone"])

    edges_to_plot = mst_edges if mst_edges else edges
    for edge in edges_to_plot:
        if edge["from"] in G.nodes() and edge["to"] in G.nodes():
            G.add_edge(edge["from"], edge["to"], weight=edge.get("cost", 0))

    pos = nx.get_node_attributes(G, 'pos')
    
    # Colores por zona
    zone_colors = {}
    colors = plt.cm.tab20(range(len(zones)))
    for i, zone in enumerate(zones):
        zone_colors[zone["name"]] = colors[i]
    
    node_colors = [zone_colors[nodes[nid-1]["zone"]] for nid in G.nodes()]
    
    # Dibujar
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=20, alpha=0.8, ax=ax)
    
    edge_color = 'red' if mst_edges else 'gray'
    edge_alpha = 0.6 if mst_edges else 0.2
    edge_width = 1.5 if mst_edges else 0.5
    
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width=edge_width, edge_color=edge_color, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Longitud", fontsize=12)
    ax.set_ylabel("Latitud", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Leyenda
    if not selected_zone:
        legend_elements = [Patch(facecolor=zone_colors[zone["name"]], label=zone["name"]) 
                          for zone in zones[:10]]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
    
    return fig

# ========== INTERFAZ STREAMLIT ==========

# Header
st.title("⚡ Red Eléctrica de Lima - Algoritmo de Kruskal (MST)")
st.markdown("### Optimización de Costos mediante Árbol de Expansión Mínima")
st.divider()

# Sidebar
with st.sidebar:
    st.header("🔧 Configuración")
    
    seed = st.number_input("Semilla aleatoria", min_value=1, max_value=1000, value=42, 
                           help="Cambia la semilla para generar diferentes grafos")
    
    st.divider()
    
    zona_filter = st.selectbox(
        "Filtrar por zona (opcional)",
        ["Todas"] + [z["name"] for z in zones]
    )
    
    st.divider()
    
    mostrar_estadisticas = st.checkbox("Mostrar estadísticas detalladas", value=True)
    mostrar_tabla = st.checkbox("Mostrar tabla de aristas MST", value=False)
    
    st.divider()
    st.markdown("### 📊 Técnicas Utilizadas")
    st.info("""
    - **MST**: Kruskal
    - **UFDS**: Union-Find
    - **Grafos**: Recorrido y optimización
    - **Complejidad**: O(E log E)
    """)

# Generar datos
nodes, edges = generar_red_electrica(zones, seed)

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🗺️ Grafo Completo", "🎯 MST Óptimo", "📊 Análisis por Zona"])

with tab1:
    st.header("Panel de Control")
    
    # Calcular MST
    with st.spinner("🔄 Calculando MST con algoritmo de Kruskal..."):
        mst_edges, mst_cost = kruskal_mst(len(nodes), edges)
        total_cost_original = sum(e['cost'] for e in edges)
        ahorro = total_cost_original - mst_cost
        porcentaje_ahorro = (ahorro / total_cost_original) * 100
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔌 Total Postes", f"{len(nodes):,}")
    
    with col2:
        st.metric("🔗 Aristas Totales", f"{len(edges):,}")
    
    with col3:
        st.metric("✅ Aristas MST", f"{len(mst_edges):,}")
    
    with col4:
        st.metric("⚙️ Zonas", len(zones))
    
    st.divider()
    
    # Costos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "💰 Costo Total Original",
            f"S/ {total_cost_original:,.2f}",
            help="Costo si se construyeran todas las conexiones"
        )
    
    with col2:
        st.metric(
            "🎯 Costo MST Óptimo",
            f"S/ {mst_cost:,.2f}",
            delta=f"-{porcentaje_ahorro:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "💎 Ahorro Total",
            f"S/ {ahorro:,.2f}",
            delta=f"{porcentaje_ahorro:.1f}%"
        )
    
    # Gráfico de comparación
    st.divider()
    st.subheader("📊 Comparación de Costos")
    
    chart_data = pd.DataFrame({
        'Categoría': ['Grafo Completo', 'MST (Kruskal)', 'Ahorro'],
        'Costo (S/)': [total_cost_original, mst_cost, ahorro]
    })
    
    st.bar_chart(chart_data.set_index('Categoría'))
    
    # Estadísticas por zona
    if mostrar_estadisticas:
        st.divider()
        st.subheader("📍 Estadísticas por Zona")
        
        zone_stats = []
        for zone in zones:
            zone_nodes_count = len([n for n in nodes if n["zone"] == zone["name"]])
            zone_mst_edges = [e for e in mst_edges 
                             if nodes[e["from"]-1]["zone"] == zone["name"] 
                             and nodes[e["to"]-1]["zone"] == zone["name"]]
            zone_cost = sum(e["cost"] for e in zone_mst_edges)
            
            zone_stats.append({
                "Zona": zone["name"],
                "Tipo": zone["type"],
                "Postes": zone_nodes_count,
                "Aristas MST": len(zone_mst_edges),
                "Costo MST (S/)": f"{zone_cost:,.2f}"
            })
        
        df_zones = pd.DataFrame(zone_stats)
        st.dataframe(df_zones, use_container_width=True, hide_index=True)

with tab2:
    st.header("🗺️ Grafo Completo - Todas las Conexiones")
    st.info("Este grafo muestra todas las conexiones posibles entre postes eléctricos.")
    
    selected = None if zona_filter == "Todas" else zona_filter
    
    with st.spinner("Generando visualización..."):
        fig = plot_graph(nodes, edges, 
                        f"Grafo Completo - {zona_filter}", 
                        selected_zone=selected)
        st.pyplot(fig)
    
    st.caption(f"Total de aristas mostradas: {len(edges):,}")

with tab3:
    st.header("🎯 Árbol de Expansión Mínima (MST) - Algoritmo de Kruskal")
    st.success(f"✅ Solución óptima calculada: **S/ {mst_cost:,.2f}** (Ahorro: {porcentaje_ahorro:.2f}%)")
    
    selected = None if zona_filter == "Todas" else zona_filter
    
    with st.spinner("Generando visualización MST..."):
        fig_mst = plot_graph(nodes, edges, 
                            f"MST - Kruskal | Costo: S/ {mst_cost:,.2f}", 
                            mst_edges=mst_edges,
                            selected_zone=selected)
        st.pyplot(fig_mst)
    
    st.caption(f"Aristas en el MST: {len(mst_edges):,} (debe ser {len(nodes)-1})")
    
    # Tabla de aristas MST
    if mostrar_tabla:
        st.divider()
        st.subheader("📋 Detalle de Aristas del MST")
        
        mst_table = []
        for i, edge in enumerate(mst_edges[:100], 1):  # Primeras 100
            mst_table.append({
                "#": i,
                "Desde": edge["from"],
                "Hasta": edge["to"],
                "Distancia (m)": edge["distance"],
                "Costo (S/)": f"{edge['cost']:.2f}"
            })
        
        df_mst = pd.DataFrame(mst_table)
        st.dataframe(df_mst, use_container_width=True, hide_index=True)
        
        if len(mst_edges) > 100:
            st.caption(f"Mostrando las primeras 100 de {len(mst_edges)} aristas")

with tab4:
    st.header("📊 Análisis Detallado por Zona")
    
    zona_analisis = st.selectbox(
        "Selecciona una zona para análisis detallado",
        [z["name"] for z in zones]
    )
    
    # Filtrar datos de la zona
    zone_nodes_ids = [n["id"] for n in nodes if n["zone"] == zona_analisis]
    zone_mst_edges = [e for e in mst_edges 
                      if e["from"] in zone_nodes_ids and e["to"] in zone_nodes_ids]
    
    # Métricas de la zona
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Postes en la zona", len(zone_nodes_ids))
    
    with col2:
        st.metric("Aristas MST", len(zone_mst_edges))
    
    with col3:
        zone_cost = sum(e["cost"] for e in zone_mst_edges)
        st.metric("Costo MST", f"S/ {zone_cost:,.2f}")
    
    # Visualización de la zona
    st.divider()
    
    with st.spinner(f"Generando visualización de {zona_analisis}..."):
        fig_zone = plot_graph(nodes, edges, 
                             f"MST - {zona_analisis}", 
                             mst_edges=mst_edges,
                             selected_zone=zona_analisis)
        st.pyplot(fig_zone)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎓 Complejidad Algorítmica - 1ACC0184 | UPC 2025-20</p>
    <p>Técnica: <b>MST (Kruskal)</b> + <b>UFDS (Union-Find)</b> | Complejidad: <b>O(E log E)</b></p>
</div>
""", unsafe_allow_html=True)
