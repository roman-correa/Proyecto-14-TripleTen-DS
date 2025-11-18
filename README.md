# 📈  Predicción de Precios de Vehículos

## Análisis y Modelado Predictivo de Precio de Vehiculos

Este proyecto corresponde al **Sprint 14** del bootcamp de Data Science de **TripleTen**. El objetivo principal es la aplicación de técnicas de **Machine Learning (Regresión)** para [DESCRIBE EL OBJETIVO CLAVE, ej: predecir un valor de mercado con alta precisión]. El análisis se encuentra documentado en el *Jupyter Notebook* `sp14.ipynb`.

## 🎯 Objetivos del Proyecto

1.  **Limpieza de Datos:** Implementar estrategias para el manejo de valores nulos (NaN) y el tratamiento de *outliers*.
2.  **Análisis Exploratorio (EDA):** Identificar la distribución, correlaciones y la influencia de las características en la variable objetivo.
3.  **Modelado:** Entrenar y optimizar modelos de **Regresión** (ej. Random Forest, LightGBM).
4.  **Evaluación:** Comparar el rendimiento de los modelos utilizando métricas clave, con énfasis en el **Error Cuadrático Medio de la Raíz ($RMSE$)** y el **Coeficiente de Determinación ($R^2$)**.

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Propósito |
| :--- | :--- |
| **Python** | Lenguaje de programación principal. |
| **Pandas & NumPy** | Manipulación y computación de datos. |
| **Scikit-learn** | Implementación de modelos y *pipelines* de ML. |
| **Matplotlib & Seaborn** | Visualizaciones para el EDA. |
| **Jupyter Notebook** | Entorno de desarrollo (`sp14.ipynb`). |

---

## 📊 Resultados Clave

### Metodología
El flujo de trabajo incluyó un preprocesamiento riguroso (codificación de categóricos, escalado/estandarización) seguido de la búsqueda de hiperparámetros.

### Rendimiento del Modelo
El modelo **[Modelo Seleccionado, ej: LightGBM Regressor]** demostró ser el más efectivo:

| Modelo | Métrica Clave [EJ: $RMSE$] | Métrica Secundaria [EJ: $R^2$] |
| :--- | :--- | :--- |
| **[Modelo Seleccionado]** | **[MEJOR VALOR]** | **[MEJOR VALOR]** |

### Conclusión
El modelo final logra predecir **[Precio]** con una desviación promedio de **[VALOR $RMSE$]**. El análisis de la importancia de las características reveló que **EMPRESA (marca)** es el factor más determinante en la predicción.

---

## 🚀 Cómo Ejecutar el Análisis

Para reproducir este proyecto, sigue estos pasos:

1.  **Clonar el Repositorio:**
    ```bash
    git clone [https://github.com/roman-correa/Proyecto-14-TripleTen-DS.git](https://github.com/roman-correa/Proyecto-14-TripleTen-DS.git)
    cd Proyecto-14-TripleTen-DS
    ```
2.  **Instalar Dependencias:**
    ```bash
    # Se asume que tienes un archivo requirements.txt, o instala manualmente:
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```
3.  **Abrir el Notebook:**
    ```bash
    jupyter notebook sp14.ipynb
    ```
    Abre el archivo `sp14.ipynb` y ejecuta las celdas en orden para seguir el análisis.

---

## 🧑‍💻 Autor

**Román Correa**
* **GitHub:** [roman-correa](https://github.com/roman-correa)
* **Linkedin:** [roman-correa](https://www.linkedin.com/in/bigcelph)
