import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Ventas Retail",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Función para cargar el modelo y datos
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load('../models/model_final.joblib')
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv('../data/processed/inferencia_df_transformado.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return None

# Función para realizar predicciones recursivas
def predecir_recursivo(df_producto, modelo, ajuste_descuento, ajuste_competencia):
    """
    Realiza predicciones día por día actualizando los lags recursivamente
    """
    df_sim = df_producto.copy()
    df_sim = df_sim.sort_values('fecha').reset_index(drop=True)
    
    # Ajustar precio_venta según descuento
    factor_descuento = 1 + (ajuste_descuento / 100)
    df_sim['precio_venta'] = df_sim['precio_base'] * factor_descuento
    
    # Ajustar precios de competencia
    factor_competencia = 1 + (ajuste_competencia / 100)
    if 'Amazon' in df_sim.columns:
        df_sim['Amazon'] = df_sim['Amazon'] * factor_competencia
    if 'Decathlon' in df_sim.columns:
        df_sim['Decathlon'] = df_sim['Decathlon'] * factor_competencia
    if 'Deporvillage' in df_sim.columns:
        df_sim['Deporvillage'] = df_sim['Deporvillage'] * factor_competencia
    
    # Recalcular precio_competencia como promedio
    cols_competencia = [col for col in ['Amazon', 'Decathlon', 'Deporvillage'] if col in df_sim.columns]
    if cols_competencia:
        df_sim['precio_competencia'] = df_sim[cols_competencia].mean(axis=1)
    
    # Recalcular descuento_pct y ratio_Precio
    df_sim['descuento_pct'] = ((df_sim['precio_base'] - df_sim['precio_venta']) / df_sim['precio_base']) * 100
    df_sim['ratio_Precio'] = df_sim['precio_venta'] / df_sim['precio_competencia']
    
    # Obtener features que espera el modelo
    feature_names = modelo.feature_names_in_
    
    predicciones = []
    
    # Predecir día por día
    for i in range(len(df_sim)):
        # Preparar features para predicción
        X_pred = df_sim.iloc[[i]][feature_names]
        
        # Realizar predicción
        pred = modelo.predict(X_pred)[0]
        pred = max(0, pred)  # No permitir ventas negativas
        predicciones.append(pred)
        
        # Actualizar lags para el siguiente día (si no es el último)
        if i < len(df_sim) - 1:
            # Guardar valores actuales de lags
            lag_actual_1 = df_sim.loc[i, 'lag_1']
            lag_actual_2 = df_sim.loc[i, 'lag_2']
            lag_actual_3 = df_sim.loc[i, 'lag_3']
            lag_actual_4 = df_sim.loc[i, 'lag_4']
            lag_actual_5 = df_sim.loc[i, 'lag_5']
            lag_actual_6 = df_sim.loc[i, 'lag_6']
            
            # Actualizar lags del siguiente día
            df_sim.loc[i + 1, 'lag_1'] = pred
            df_sim.loc[i + 1, 'lag_2'] = lag_actual_1
            df_sim.loc[i + 1, 'lag_3'] = lag_actual_2
            df_sim.loc[i + 1, 'lag_4'] = lag_actual_3
            df_sim.loc[i + 1, 'lag_5'] = lag_actual_4
            df_sim.loc[i + 1, 'lag_6'] = lag_actual_5
            df_sim.loc[i + 1, 'lag_7'] = lag_actual_6
            
            # Actualizar media móvil
            ultimas_predicciones = predicciones[-7:] if len(predicciones) >= 7 else predicciones
            df_sim.loc[i + 1, 'media_movil_7d'] = np.mean(ultimas_predicciones)
    
    df_sim['unidades_predichas'] = predicciones
    df_sim['ingresos'] = df_sim['unidades_predichas'] * df_sim['precio_venta']
    
    return df_sim

# Cargar modelo y datos
modelo = cargar_modelo()
df_completo = cargar_datos()

if modelo is None or df_completo is None:
    st.stop()

# Obtener lista única de productos
productos_unicos = sorted(df_completo['nombre_producto'].unique())

# ============= SIDEBAR =============
st.sidebar.markdown("## 🎮 Controles de Simulación")
st.sidebar.markdown("---")

# Selector de producto
producto_seleccionado = st.sidebar.selectbox(
    "📦 Selecciona un producto:",
    productos_unicos
)

# Slider de descuento
ajuste_descuento = st.sidebar.slider(
    "💰 Ajuste de Descuento (%)",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    help="Ajusta el descuento del precio base"
)

# Selector de escenario de competencia
st.sidebar.markdown("### 🏪 Escenario de Competencia")
escenario_competencia = st.sidebar.radio(
    "Selecciona el escenario:",
    ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    help="Simula cambios en los precios de la competencia"
)

# Mapear escenario a valor numérico
mapeo_escenario = {
    "Actual (0%)": 0,
    "Competencia -5%": -5,
    "Competencia +5%": 5
}
ajuste_competencia_principal = mapeo_escenario[escenario_competencia]

st.sidebar.markdown("---")

# Botón de simulación
simular = st.sidebar.button("🚀 Simular Ventas", type="primary", use_container_width=True)

# ============= ZONA PRINCIPAL =============

# Header
st.markdown(f"""
    <div style='background-color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
        <h1 style='color: #667eea; margin: 0;'>📊 Dashboard de Predicción de Ventas</h1>
        <p style='color: #666; font-size: 18px; margin: 10px 0 0 0;'>Noviembre 2025 - {producto_seleccionado}</p>
    </div>
""", unsafe_allow_html=True)

if simular:
    with st.spinner('🔮 Realizando predicciones recursivas...'):
        # Filtrar datos del producto seleccionado
        df_producto = df_completo[df_completo['nombre_producto'] == producto_seleccionado].copy()
        
        if len(df_producto) == 0:
            st.error("❌ No se encontraron datos para el producto seleccionado")
            st.stop()
        
        # Realizar predicción con parámetros del usuario
        df_resultado = predecir_recursivo(
            df_producto, 
            modelo, 
            ajuste_descuento, 
            ajuste_competencia_principal
        )
        
        # Calcular KPIs
        unidades_totales = df_resultado['unidades_predichas'].sum()
        ingresos_totales = df_resultado['ingresos'].sum()
        precio_promedio = df_resultado['precio_venta'].mean()
        descuento_promedio = df_resultado['descuento_pct'].mean()
        
        # ============= KPIs DESTACADOS =============
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 Unidades Totales",
                value=f"{int(unidades_totales):,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="💰 Ingresos Proyectados",
                value=f"€{ingresos_totales:,.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="💵 Precio Promedio",
                value=f"€{precio_promedio:.2f}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="🏷️ Descuento Promedio",
                value=f"{descuento_promedio:.1f}%",
                delta=None
            )
        
        st.markdown("---")
        
        # ============= GRÁFICO DE PREDICCIÓN DIARIA =============
        st.markdown("### 📈 Evolución de Ventas Diarias - Noviembre 2025")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Configurar estilo de seaborn
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Crear gráfico de línea
        dias = df_resultado['dia_mes'].values
        ventas = df_resultado['unidades_predichas'].values
        
        # Línea principal
        sns.lineplot(x=dias, y=ventas, marker='o', linewidth=2.5, markersize=6, 
                    color='#667eea', ax=ax)
        
        # Identificar Black Friday (día 28)
        idx_bf = df_resultado[df_resultado['dia_mes'] == 28].index[0]
        dia_bf = df_resultado.loc[idx_bf, 'dia_mes']
        ventas_bf = df_resultado.loc[idx_bf, 'unidades_predichas']
        
        # Marcar Black Friday
        ax.axvline(x=dia_bf, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                  label='Black Friday')
        ax.scatter([dia_bf], [ventas_bf], color='red', s=200, zorder=5, 
                  edgecolors='darkred', linewidth=2)
        ax.annotate('🛍️ BLACK FRIDAY', 
                   xy=(dia_bf, ventas_bf), 
                   xytext=(dia_bf + 1, ventas_bf * 1.1),
                   fontsize=12,
                   fontweight='bold',
                   color='red',
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_xlabel('Día del Mes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
        ax.set_title('Predicción de Ventas Diarias', fontsize=14, fontweight='bold', 
                    color='#667eea')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper left')
        
        # Mejorar visualización del eje X
        ax.set_xticks(range(1, 31, 2))
        
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # ============= TABLA DETALLADA =============
        st.markdown("### 📋 Detalle Diario de Predicciones")
        
        # Preparar tabla
        df_tabla = df_resultado[['fecha', 'nombre_dia', 'precio_venta', 'precio_competencia', 
                                 'descuento_pct', 'unidades_predichas', 'ingresos']].copy()
        
        df_tabla['fecha'] = df_tabla['fecha'].dt.strftime('%Y-%m-%d')
        df_tabla['precio_venta'] = df_tabla['precio_venta'].apply(lambda x: f"€{x:.2f}")
        df_tabla['precio_competencia'] = df_tabla['precio_competencia'].apply(lambda x: f"€{x:.2f}")
        df_tabla['descuento_pct'] = df_tabla['descuento_pct'].apply(lambda x: f"{x:.1f}%")
        df_tabla['unidades_predichas'] = df_tabla['unidades_predichas'].apply(lambda x: f"{int(x)}")
        df_tabla['ingresos'] = df_tabla['ingresos'].apply(lambda x: f"€{x:.2f}")
        
        # Añadir emoji para Black Friday
        df_tabla.loc[df_tabla.index == idx_bf, 'nombre_dia'] = '🛍️ ' + df_tabla.loc[idx_bf, 'nombre_dia']
        
        df_tabla.columns = ['Fecha', 'Día', 'Precio Venta', 'Precio Competencia', 
                           'Descuento', 'Unidades', 'Ingresos']
        
        # Mostrar tabla con estilo
        st.dataframe(
            df_tabla,
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # ============= COMPARATIVA DE ESCENARIOS =============
        st.markdown("### 🔄 Comparativa de Escenarios de Competencia")
        st.markdown("*Análisis de impacto según precios de la competencia (manteniendo tu descuento actual)*")
        
        with st.spinner('📊 Comparando escenarios...'):
            # Simular los 3 escenarios
            escenarios = {
                "Sin Cambios (0%)": 0,
                "Competencia Baja (-5%)": -5,
                "Competencia Alta (+5%)": 5
            }
            
            resultados_escenarios = {}
            
            for nombre_esc, ajuste_comp in escenarios.items():
                df_esc = predecir_recursivo(df_producto, modelo, ajuste_descuento, ajuste_comp)
                resultados_escenarios[nombre_esc] = {
                    'unidades': df_esc['unidades_predichas'].sum(),
                    'ingresos': df_esc['ingresos'].sum()
                }
            
            # Mostrar resultados
            col1, col2, col3 = st.columns(3)
            
            for i, (nombre_esc, resultados) in enumerate(resultados_escenarios.items()):
                col = [col1, col2, col3][i]
                
                emoji = "📊" if "0%" in nombre_esc else ("📉" if "-5%" in nombre_esc else "📈")
                
                with col:
                    st.markdown(f"""
                        <div style='background-color: white; padding: 20px; border-radius: 10px; 
                                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center;'>
                            <h3 style='color: #667eea; margin-bottom: 15px;'>{emoji} {nombre_esc}</h3>
                            <p style='font-size: 24px; font-weight: bold; color: #333; margin: 10px 0;'>
                                {int(resultados['unidades']):,}
                            </p>
                            <p style='color: #666; margin: 5px 0;'>Unidades</p>
                            <hr style='margin: 15px 0;'>
                            <p style='font-size: 20px; font-weight: bold; color: #667eea; margin: 10px 0;'>
                                €{resultados['ingresos']:,.2f}
                            </p>
                            <p style='color: #666; margin: 5px 0;'>Ingresos</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.success("✅ Simulación completada exitosamente")

else:
    # Mensaje inicial
    st.info("👈 Configura los parámetros en el panel lateral y pulsa '🚀 Simular Ventas' para comenzar")
    
    # Mostrar información del dataset
    st.markdown("### 📊 Información del Dataset")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Productos Disponibles", len(productos_unicos))
    
    with col2:
        st.metric("Días a Predecir", "30")
    
    with col3:
        st.metric("Mes de Predicción", "Noviembre 2025")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        <p>🤖 Simulador de Ventas Retail | Powered by HistGradientBoostingRegressor</p>
    </div>
""", unsafe_allow_html=True)