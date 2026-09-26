"""Genera notebooks/Laboratorio7_Spark.ipynb (secciones 1-4).

Uso:  python scripts/build_notebook.py
"""
import json
from pathlib import Path

CELLS = []


def md(src):
    CELLS.append(("markdown", src.strip("\n")))


def code(src):
    CELLS.append(("code", src.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Laboratorio 7 — Spark MLlib
## Encuesta Nacional de Empleo e Ingresos Continua (ENEIC) — Personas

**Parte I: Análisis exploratorio avanzado y segmentación (25 pts)**

1. Carga, armonización y calidad de datos
2. Estadística descriptiva y preguntas de exploración
3. Relaciones entre variables numéricas
4. Segmentación de perfiles mediante KMeans

> El notebook está pensado para ejecutarse de principio a fin (*Run All*). Los archivos Excel originales deben estar en `data/raw/` con los nombres configurados en la celda de configuración.
> Todas las estadísticas y métricas se calculan en Spark sobre el conjunto completo; a pandas solo se transfieren tablas agregadas o muestras de hasta 10,000 registros para graficar.
""")

md("## 0. Configuración del entorno")

code(r"""
import math
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = (
    SparkSession.builder
    .appName("Lab7-ENEIC")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.session.timeZone", "America/Guatemala")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

SEED = 42
N_MUESTRA_GRAF = 10_000          # tamaño máximo de las muestras que se llevan a pandas
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

print("Spark", spark.version)
""")

md(r"""
### Rutas y archivos

Cada archivo se identifica **por su procedencia** (no por la columna `TRIMESTRE`). Ajuste `archivo` si sus nombres locales son distintos.
""")

code(r"""
DATA_RAW = Path("../data/raw")
DATA_PROC = Path("../data/processed")
DATA_PROC.mkdir(parents=True, exist_ok=True)

ARCHIVOS = [
    {"periodo_archivo": "2025T1", "anio_archivo": 2025, "trimestre_calendario": 1, "uso": "desarrollo", "archivo": "ENEIC_2025_T1_Personas.xlsx"},
    {"periodo_archivo": "2025T2", "anio_archivo": 2025, "trimestre_calendario": 2, "uso": "desarrollo", "archivo": "ENEIC_2025_T2_Personas.xlsx"},
    {"periodo_archivo": "2025T3", "anio_archivo": 2025, "trimestre_calendario": 3, "uso": "desarrollo", "archivo": "ENEIC_2025_T3_Personas.xlsx"},
    {"periodo_archivo": "2025T4", "anio_archivo": 2025, "trimestre_calendario": 4, "uso": "validacion", "archivo": "ENEIC_2025_T4_Personas.xlsx"},
    {"periodo_archivo": "2026T1", "anio_archivo": 2026, "trimestre_calendario": 1, "uso": "prueba",     "archivo": "ENEIC_2026_T1_Personas.xlsx"},
]

faltantes_arch = [a["archivo"] for a in ARCHIVOS if not (DATA_RAW / a["archivo"]).exists()]
assert not faltantes_arch, f"No se encontraron en {DATA_RAW.resolve()}: {faltantes_arch}"
""")

md(r"""
### Variables y diccionario de códigos

Los códigos categóricos se validan contra el diccionario de datos de la ENEIC. Cualquier valor ausente o no reconocido se representa como `DESCONOCIDO` (nunca como 0). En educación, el código **0 = Ninguno** es una respuesta válida.

> ⚠️ Verifique que las etiquetas de `NIVEL_EDUCATIVO` y `DOMINIO` coincidan con el diccionario entregado con cada base. Más abajo se imprimen los códigos observados para confirmarlo.
""")

code(r"""
COLS_ORIGINALES = [
    "NUM_HOGAR", "NUM_PERSONA", "FACTOR", "ANIO", "TRIMESTRE",
    "DOMINIO", "OCUPADOS", "P02A03", "P03A03A", "P05C07A", "P05C07B",
    "P05C16", "P05H01A", "P05D01",
]

CATEGORIA_OCUPACIONAL = {
    1: "Empleado de gobierno",
    2: "Empleado de empresa privada",
    3: "Jornalero o peón",
    4: "Servicio doméstico",
}
NIVEL_EDUCATIVO = {      # P03A03A — nivel más alto aprobado
    0: "Ninguno",
    1: "Preprimaria",
    2: "Primaria",
    3: "Básico",
    4: "Diversificado",
    5: "Superior",
    6: "Postgrado",
}
DOMINIO = {
    1: "Urbano metropolitano",
    2: "Resto urbano",
    3: "Rural nacional",
}
ORDEN_EDU = list(NIVEL_EDUCATIVO.values()) + ["DESCONOCIDO"]
""")

# ---------------------------------------------------------------------------
md(r"""
---
# 1. Carga, armonización y calidad de datos

### 1.1 Lectura individual de cada Excel

Spark no lee Excel de forma nativa. Cada archivo se procesa **por separado** con pandas/openpyxl, leyendo solo las columnas requeridas (`usecols`) para controlar la memoria. Los encabezados se normalizan a mayúsculas y cada valor se lleva a una representación de texto consistente (`3`, `3.0` y `"3 "` → `"3"`), de modo que un mismo código llegue igual aunque en un archivo sea número y en otro texto. Luego se crea un DataFrame de Spark con **esquema explícito** y se guarda en Parquet.
""")

code(r"""
def normalizar_valor(v):
    # Representación textual consistente de una celda de Excel (None = ausente).
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return str(int(v)) if float(v).is_integer() else repr(float(v))
    s = str(v).strip()
    if s == "":
        return None
    try:
        f = float(s)
        if math.isfinite(f) and f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s


ESQUEMA_CRUDO = T.StructType(
    [T.StructField(c, T.StringType(), True) for c in COLS_ORIGINALES]
    + [
        T.StructField("archivo_origen", T.StringType(), False),
        T.StructField("periodo_archivo", T.StringType(), False),
        T.StructField("anio_archivo", T.IntegerType(), False),
        T.StructField("trimestre_calendario", T.IntegerType(), False),
    ]
)


def excel_a_parquet(info):
    ruta = DATA_RAW / info["archivo"]
    encabezado = pd.read_excel(ruta, nrows=0, engine="openpyxl")
    mapa = {c: str(c).strip().upper() for c in encabezado.columns}
    usar = [c for c, u in mapa.items() if u in COLS_ORIGINALES]
    ausentes = sorted(set(COLS_ORIGINALES) - {mapa[c] for c in usar})

    pdf = pd.read_excel(ruta, usecols=usar, dtype=object, engine="openpyxl")
    pdf.columns = [mapa[c] for c in pdf.columns]
    for c in ausentes:
        pdf[c] = None
    pdf = pdf[COLS_ORIGINALES]

    meta = (info["archivo"], info["periodo_archivo"], info["anio_archivo"], info["trimestre_calendario"])
    filas = [tuple(normalizar_valor(v) for v in fila) + meta for fila in pdf.itertuples(index=False, name=None)]
    sdf = spark.createDataFrame(filas, schema=ESQUEMA_CRUDO)

    destino = DATA_PROC / "crudo" / info["periodo_archivo"]
    sdf.write.mode("overwrite").parquet(str(destino))
    resumen = {
        "periodo_archivo": info["periodo_archivo"],
        "archivo_origen": info["archivo"],
        "filas_originales": len(pdf),
        "columnas_originales": len(encabezado.columns),
        "columnas_requeridas_ausentes": ", ".join(ausentes) or "—",
    }
    del pdf, filas
    return resumen


resumen_carga = pd.DataFrame([excel_a_parquet(a) for a in ARCHIVOS])
resumen_carga
""")

md(r"""
**Comentario.** La tabla anterior permite contrastar con los conteos oficiales (51,588; 51,167; 51,583; 49,338 y 49,843 registros). El archivo de IV‑2025 tiene **302 columnas** frente a 270 del resto; por eso las columnas se seleccionan y unen **por nombre**.
""")

md("### 1.2 Unión de los archivos de 2025 con `unionByName` y auditoría de `TRIMESTRE`")

code(r"""
crudos = {a["periodo_archivo"]: spark.read.parquet(str(DATA_PROC / "crudo" / a["periodo_archivo"])) for a in ARCHIVOS}

PERIODOS_2025 = [a["periodo_archivo"] for a in ARCHIVOS if a["anio_archivo"] == 2025]
crudo_2025 = reduce(lambda a, b: a.unionByName(b), [crudos[p] for p in PERIODOS_2025])
crudo_2026 = crudos["2026T1"]
crudo_todo = crudo_2025.unionByName(crudo_2026)

print("Registros 2025 (4 archivos):", f"{crudo_2025.count():,}")
print("Registros 2026T1:", f"{crudo_2026.count():,}")

# Auditoría: valores originales de TRIMESTRE por archivo (se conserva, no se usa como trimestre calendario)
(crudo_todo.groupBy("periodo_archivo", "ANIO", "TRIMESTRE").count()
 .orderBy("periodo_archivo", "TRIMESTRE").toPandas())
""")

md(r"""
Como indica el enunciado, en II‑2025 aparecen registros con `TRIMESTRE = 2`. Restar 1 a `TRIMESTRE` asignaría esos registros a un trimestre incorrecto; por eso `periodo_archivo`, `anio_archivo` y `trimestre_calendario` se derivan del **archivo de procedencia** y `archivo_origen` preserva la trazabilidad.
""")

md(r"""
### 1.3 Homologación de tipos

- Numéricas (`edad`, antigüedad, horas, salario, `FACTOR`) → `double`. Un texto no numérico se convierte en nulo y luego se contabiliza como no evaluable.
- Códigos (`OCUPADOS`, `P05C16`, `P03A03A`, `DOMINIO`) → entero solo si el valor es numérico entero; después se mapean a etiquetas del diccionario o a `DESCONOCIDO`.
- Identificadores `NUM_HOGAR`, `NUM_PERSONA` → `long`; `ANIO`, `TRIMESTRE` → `int` (solo auditoría).
""")

code(r"""
INF = float("inf")


def a_double(c):
    return F.col(c).cast("double")


def a_codigo(c):
    d = F.col(c).cast("double")
    return F.when(d.isNotNull() & ~F.isnan(d) & (F.abs(d) != INF) & (d == F.floor(d)), d.cast("int"))


def etiqueta(col_codigo, diccionario):
    mapa = F.create_map(*[x for k, v in diccionario.items() for x in (F.lit(k), F.lit(v))])
    return F.coalesce(mapa[F.col(col_codigo)], F.lit("DESCONOCIDO"))


def tipar(df):
    return (
        df.select(
            "archivo_origen", "periodo_archivo", "anio_archivo", "trimestre_calendario",
            F.col("NUM_HOGAR").cast("long").alias("NUM_HOGAR"),
            F.col("NUM_PERSONA").cast("long").alias("NUM_PERSONA"),
            a_double("FACTOR").alias("FACTOR"),
            a_codigo("ANIO").alias("ANIO"),
            a_codigo("TRIMESTRE").alias("TRIMESTRE"),
            a_codigo("OCUPADOS").alias("ocupado"),
            a_double("P02A03").alias("edad"),
            a_double("P05C07A").alias("antiguedad_anios"),
            a_double("P05C07B").alias("antiguedad_meses"),
            a_double("P05H01A").alias("horas_semanales"),
            a_double("P05D01").alias("salario_mensual"),
            a_codigo("P05C16").alias("categoria_ocupacional_cod"),
            a_codigo("P03A03A").alias("nivel_educativo_cod"),
            a_codigo("DOMINIO").alias("dominio_cod"),
        )
        .withColumn("antiguedad", F.col("antiguedad_anios") + F.col("antiguedad_meses") / 12.0)
        .withColumn("categoria_ocupacional", etiqueta("categoria_ocupacional_cod", CATEGORIA_OCUPACIONAL))
        .withColumn("nivel_educativo", etiqueta("nivel_educativo_cod", NIVEL_EDUCATIVO))
        .withColumn("dominio", etiqueta("dominio_cod", DOMINIO))
    )


tip_2025 = tipar(crudo_2025)
tip_2026 = tipar(crudo_2026)
tip_2025.printSchema()
""")

md("**Esquema y cinco registros de las columnas seleccionadas** (ya con nombres analíticos):")

code(r"""
COLS_MOSTRAR = ["periodo_archivo", "NUM_HOGAR", "NUM_PERSONA", "TRIMESTRE", "edad", "antiguedad_anios",
                "antiguedad_meses", "antiguedad", "horas_semanales", "salario_mensual", "ocupado",
                "categoria_ocupacional", "nivel_educativo", "dominio", "FACTOR"]
tip_2025.select(COLS_MOSTRAR).show(5, truncate=False)
""")

md("Códigos categóricos observados (para confirmar el diccionario y detectar valores no reconocidos):")

code(r"""
tip_todo = tip_2025.unionByName(tip_2026)
for cod, etq in [("nivel_educativo_cod", "nivel_educativo"), ("dominio_cod", "dominio"),
                 ("categoria_ocupacional_cod", "categoria_ocupacional"), ("ocupado", None)]:
    cols = [cod] + ([etq] if etq else [])
    print(f"--- {cod}")
    tip_todo.groupBy(*cols).count().orderBy(cod).show(30, truncate=False)
""")

md(r"""
### 1.4 Faltantes por variable **antes** de aplicar filtros

Se consideran faltantes las celdas vacías en el archivo original (texto nulo tras la normalización). Se reportan para 2025 (los cuatro archivos unidos) y para 2026T1.
""")

code(r"""
def tabla_faltantes(df, etiqueta_):
    n = df.count()
    fila = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in COLS_ORIGINALES]).first().asDict()
    return pd.DataFrame({
        "variable": COLS_ORIGINALES,
        f"faltantes_{etiqueta_}": [fila[c] for c in COLS_ORIGINALES],
        f"pct_{etiqueta_}": [100 * fila[c] / n for c in COLS_ORIGINALES],
    }).set_index("variable")


faltantes = tabla_faltantes(crudo_2025, "2025").join(tabla_faltantes(crudo_2026, "2026T1"))
faltantes
""")

md(r"""
Para distinguir entre un faltante **estructural** (la pregunta no corresponde) y una **no respuesta**, comparamos el faltante del salario y de las variables laborales en toda la base frente al subconjunto al que *sí* se le debía preguntar (15+ años, ocupados y asalariados):
""")

code(r"""
debe_responder = (
    (F.col("P02A03").cast("double") >= 15)
    & (F.col("OCUPADOS").cast("double") == 1)
    & F.col("P05C16").cast("double").isin(1, 2, 3, 4)
)
vars_lab = {"P05D01": "salario_mensual", "P05C07A": "antiguedad_anios", "P05C07B": "antiguedad_meses", "P05H01A": "horas_semanales"}
(crudo_todo
 .groupBy(F.coalesce(debe_responder, F.lit(False)).alias("asalariado_15mas"))
 .agg(F.count("*").alias("registros"),
      *[F.round(100 * F.avg(F.col(c).isNull().cast("int")), 2).alias(f"pct_nulo_{v}") for c, v in vars_lab.items()])
 .toPandas())
""")

md(r"""
La gran mayoría de los nulos de salario, antigüedad y horas se concentran en personas a quienes la pregunta **no les corresponde** (menores de 15, desocupados, inactivos o no asalariados). Los nulos que persisten dentro de los asalariados de 15+ son **no respuesta** real.
""")

md(r"""
### 1.5 Filtros de población y de calidad (orden fijo)

Cada registro recibe el **primer** motivo por el que se excluye, siempre en el mismo orden:

| Paso | Criterio para conservar |
|---|---|
| 1 | Edad finita y ≥ 15 |
| 2 | `OCUPADOS = 1` |
| 3 | `P05C16` ∈ {1, 2, 3, 4} (asalariado) |
| 4 | Salario `P05D01` numérico, finito y > 0 (no se imputa) |
| 5 | Años de antigüedad finitos y ≥ 0 |
| 6 | Meses de antigüedad enteros entre 0 y 11 |
| 7 | Antigüedad calculada ≤ edad |
| 8 | Horas habituales > 0 y ≤ 168 |

Un valor nulo en cualquiera de las variables del criterio implica que el registro **no puede evaluarse** y se excluye en ese paso.
""")

code(r"""
def finito(c):
    return F.col(c).isNotNull() & ~F.isnan(c) & (F.abs(F.col(c)) != INF)


PASOS = [
    ("1_edad_menor_15_o_invalida", finito("edad") & (F.col("edad") >= 15)),
    ("2_no_ocupado", F.col("ocupado") == 1),
    ("3_no_asalariado", F.col("categoria_ocupacional_cod").isin(1, 2, 3, 4)),
    ("4_salario_no_positivo_o_faltante", finito("salario_mensual") & (F.col("salario_mensual") > 0)),
    ("5_antig_anios_invalida", finito("antiguedad_anios") & (F.col("antiguedad_anios") >= 0)),
    ("6_antig_meses_invalida", finito("antiguedad_meses") & (F.col("antiguedad_meses") == F.floor("antiguedad_meses"))
                               & F.col("antiguedad_meses").between(0, 11)),
    ("7_antiguedad_mayor_que_edad", F.col("antiguedad") <= F.col("edad")),
    ("8_horas_fuera_de_rango", finito("horas_semanales") & (F.col("horas_semanales") > 0) & (F.col("horas_semanales") <= 168)),
]
NOMBRES_PASOS = [p for p, _ in PASOS]


def asignar_motivo(df):
    motivo = None
    for nombre, condicion in PASOS:
        cumple = F.coalesce(condicion, F.lit(False))
        motivo = F.when(~cumple, F.lit(nombre)) if motivo is None else motivo.when(~cumple, F.lit(nombre))
    return df.withColumn("motivo_exclusion", motivo.otherwise(F.lit(None)))


marcado_2025 = asignar_motivo(tip_2025)
marcado_2026 = asignar_motivo(tip_2026)
marcado_todo = marcado_2025.unionByName(marcado_2026).cache()

conteo = (marcado_todo.groupBy("periodo_archivo", F.coalesce("motivo_exclusion", F.lit("conservado")).alias("motivo"))
          .count().toPandas()
          .pivot(index="motivo", columns="periodo_archivo", values="count").fillna(0).astype(int))
conteo = conteo.reindex(NOMBRES_PASOS + ["conservado"]).fillna(0).astype(int)
conteo["Total 2025"] = conteo[PERIODOS_2025].sum(axis=1)

cascada = pd.concat([
    conteo.sum().to_frame("0_registros_originales").T,
    conteo.loc[NOMBRES_PASOS].rename(index=lambda s: f"excluidos_{s}"),
    conteo.loc[["conservado"]].rename(index={"conservado": "registros_despues_de_filtros"}),
])
cascada.loc["pct_conservado"] = 100 * cascada.loc["registros_despues_de_filtros"] / cascada.loc["0_registros_originales"]
cascada
""")

code(r"""
restantes = cascada.loc["0_registros_originales"] - cascada.loc[[f"excluidos_{p}" for p in NOMBRES_PASOS]].cumsum()
fig, ax = plt.subplots(figsize=(10, 4.5))
for p in PERIODOS_2025 + ["2026T1"]:
    ax.plot(["0_origen"] + NOMBRES_PASOS, [cascada.loc["0_registros_originales", p]] + list(restantes[p]), marker="o", label=p)
ax.set_title("Registros restantes después de cada paso de filtrado")
ax.set_ylabel("Registros")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
ax.tick_params(axis="x", rotation=35)
plt.setp(ax.get_xticklabels(), ha="right")
ax.legend()
plt.tight_layout(); plt.show()
""")

md(r"""
**Interpretación.** Los pasos 1 a 3 definen la *población* (asalariados de 15+ años) y explican la mayor parte de las exclusiones: en una encuesta de hogares la mayoría de personas son menores, inactivas, desocupadas o trabajadores no asalariados (cuenta propia, patronos, familiares no remunerados). Los pasos 4 a 8 son de *calidad*: eliminan registros sin salario positivo o con antigüedad/horas no evaluables o incoherentes. Ninguna variable objetivo se imputa.
""")

md(r"""
### 1.6 Unicidad de `periodo_archivo + NUM_HOGAR + NUM_PERSONA`

La verificación se hace sobre los datos **crudos** (todas las filas) y sobre la base **filtrada**. Si existe alguna clave repetida se clasifica como:

- **Repetición exacta**: todas las columnas originales seleccionadas coinciden.
- **Registro en conflicto**: misma clave pero valores distintos en alguna columna.
""")

code(r"""
CLAVE = ["periodo_archivo", "NUM_HOGAR", "NUM_PERSONA"]


def auditar_clave(df, nombre):
    total = df.count()
    distintas = df.select(CLAVE).distinct().count()
    nulas = df.filter(F.col("NUM_HOGAR").isNull() | F.col("NUM_PERSONA").isNull()).count()
    print(f"{nombre}: {total:,} filas | {distintas:,} claves distintas | {total - distintas:,} filas sobrantes | {nulas:,} con clave nula")


auditar_clave(crudo_todo, "Crudo (5 archivos)")
auditar_clave(marcado_todo.filter(F.col("motivo_exclusion").isNull()), "Filtrado (5 archivos)")

cols_contenido = [c for c in COLS_ORIGINALES if c not in ("NUM_HOGAR", "NUM_PERSONA")]
dup = (crudo_todo
       .withColumn("firma", F.sha2(F.concat_ws("|", *[F.coalesce(F.col(c), F.lit("∅")) for c in cols_contenido]), 256))
       .groupBy(CLAVE)
       .agg(F.count("*").alias("n_filas"), F.countDistinct("firma").alias("n_versiones"))
       .filter("n_filas > 1")
       .withColumn("tipo", F.when(F.col("n_versiones") == 1, "repeticion_exacta").otherwise("conflicto")))

resumen_dup = dup.groupBy("periodo_archivo", "tipo").agg(F.count("*").alias("claves"), F.sum("n_filas").alias("filas")).toPandas()
resumen_dup if len(resumen_dup) else print("No hay claves duplicadas dentro de ningún período.")
""")

code(r"""
# Detalle de hasta 10 claves duplicadas (si las hay) para inspeccionar en qué columnas difieren
if len(resumen_dup):
    ejemplos = dup.orderBy(F.desc("n_versiones")).limit(10)
    crudo_todo.join(ejemplos.select(CLAVE + ["tipo"]), CLAVE).orderBy(CLAVE).show(40, truncate=False)
""")

md(r"""
**Decisión.** No se usa `dropDuplicates()`. Si aparecen claves repetidas, se documentan arriba: una *repetición exacta* indica una fila copiada en la fuente, mientras que un *conflicto* indica que la clave no identifica a una sola persona en ese corte (por ejemplo, una numeración de hogar que se reinicia por región o un error de captura). Ambos casos se marcan con la columna `clave_duplicada` en la base preparada para que el análisis posterior pueda evaluar su efecto. Una misma persona en **periodos distintos** no es un duplicado.
""")

md(r"""
### 1.7 Base preparada y guardado en Parquet
""")

code(r"""
claves_dup = dup.select(CLAVE).withColumn("clave_duplicada", F.lit(True))

COLS_ANALITICAS = [
    "archivo_origen", "periodo_archivo", "anio_archivo", "trimestre_calendario",
    "NUM_HOGAR", "NUM_PERSONA", "FACTOR", "ANIO", "TRIMESTRE",
    "salario_mensual", "edad", "antiguedad_anios", "antiguedad_meses", "antiguedad", "horas_semanales",
    "nivel_educativo_cod", "nivel_educativo", "categoria_ocupacional_cod", "categoria_ocupacional",
    "dominio_cod", "dominio", "ocupado",
]


def preparar(marcado):
    return (marcado.filter(F.col("motivo_exclusion").isNull())
            .join(F.broadcast(claves_dup), CLAVE, "left")
            .withColumn("clave_duplicada", F.coalesce("clave_duplicada", F.lit(False)))
            .select(COLS_ANALITICAS + ["clave_duplicada"]))


preparado_2025 = preparar(marcado_2025)
preparado_2026 = preparar(marcado_2026)
preparado_2025.write.mode("overwrite").partitionBy("periodo_archivo").parquet(str(DATA_PROC / "eneic_2025_preparado.parquet"))
preparado_2026.write.mode("overwrite").parquet(str(DATA_PROC / "eneic_2026T1_preparado.parquet"))

df25 = spark.read.parquet(str(DATA_PROC / "eneic_2025_preparado.parquet")).cache()
df26 = spark.read.parquet(str(DATA_PROC / "eneic_2026T1_preparado.parquet")).cache()
print(f"2025 preparado: {df25.count():,} registros | 2026T1 preparado: {df26.count():,} registros")
df25.groupBy("periodo_archivo").count().orderBy("periodo_archivo").show()
""")

md(r"""
### 1.8 Preguntas

**¿Por qué IV de 2025 no puede apilarse por posición de columnas con los otros archivos?**
Porque tiene **302 columnas** frente a las 270 de los demás: incorpora variables nuevas que desplazan el orden. Un `union()` posicional pegaría la columna *i* de un archivo con la columna *i* de otro aunque signifiquen cosas distintas (por ejemplo, un código de educación quedaría bajo el nombre de salario) sin generar ningún error, y los tipos podrían coincidir por casualidad. `unionByName` empareja las columnas por su **nombre**, lo que garantiza que cada variable se combine con su equivalente, independientemente de su posición. Además, antes de unir se normalizan los encabezados y se seleccionan solo las columnas requeridas.

**¿Qué diferencia existe entre un dato ausente porque la pregunta no corresponde y una respuesta no registrada?**
Un ausente *porque no corresponde* es **estructural**: el cuestionario tiene saltos (filtros) y, por diseño, a un menor de edad, a un inactivo o a un trabajador por cuenta propia no se le pregunta el sueldo de asalariado. No es información perdida; la variable simplemente no aplica y esas personas quedan fuera de la población analítica. Una *respuesta no registrada* es **no respuesta**: la pregunta sí aplicaba (asalariado de 15+ años) pero el dato no se capturó (se negó, no sabía, error de captura). Solo esta última es un problema de calidad que puede sesgar resultados. La tabla de la sección 1.4 lo muestra: casi todos los nulos de salario aparecen fuera de la población asalariada.

**¿Por qué una persona observada en dos períodos no debe eliminarse como duplicado del conjunto longitudinal?**
La ENEIC es un panel rotativo: un hogar permanece varios trimestres en la muestra. Cada fila representa a una **persona en un período**, con su edad, salario, horas y antigüedad de *ese* trimestre. Dos observaciones de la misma persona en trimestres distintos son mediciones diferentes y legítimas, no copias. Eliminarlas reduciría artificialmente la muestra de ciertos trimestres, rompería la comparabilidad entre cortes y sesgaría los resultados hacia los hogares que rotan antes. Por eso la unicidad se verifica **dentro** de cada `periodo_archivo`. (Para la validación de los modelos sí conviene recordar que las observaciones repetidas no son independientes.)

**¿Por qué el número de registros de la base filtrada no representa a todos los trabajadores del país?**
1. Es una **muestra**, no un censo: cada registro representa a muchas personas según su `FACTOR` de expansión, y aquí se trabaja sin ponderar.
2. La población está **restringida** a asalariados de 15+ años con salario positivo registrado; excluye trabajadores por cuenta propia, patronos, familiares no remunerados y a quienes no reportaron salario (la no respuesta puede no ser aleatoria, p. ej. en salarios altos).
3. Al unir cuatro trimestres, **una persona puede aparecer varias veces**, así que el número de filas no equivale a personas distintas.
4. Se excluyeron registros por reglas de calidad.

El `FACTOR` se conservó porque, en un análisis poblacional, se usaría como peso para expandir la muestra (totales, medias y percentiles ponderados) y, junto con el diseño muestral (estratos y UPM), para calcular errores estándar correctos. En este laboratorio los resultados describen **los registros analizados**, no estimaciones oficiales.
""")

# ---------------------------------------------------------------------------
md(r"""
---
# 2. Estadística descriptiva y preguntas de exploración

Todos los estadísticos se calculan en Spark sobre **toda** la población analítica de 2025. Los percentiles se obtienen con la función exacta `percentile` de Spark SQL.
""")

code(r"""
VARS_NUM = ["salario_mensual", "edad", "antiguedad", "horas_semanales"]


def resumen_num(df, cols):
    filas = []
    for c in cols:
        r = df.agg(
            F.count(c).alias("n"), F.mean(c).alias("media"), F.stddev(c).alias("desv_est"),
            F.min(c).alias("min"), F.max(c).alias("max"),
            F.expr(f"percentile({c}, array(0.25, 0.5, 0.75, 0.95))").alias("p"),
            F.skewness(c).alias("asimetria"),
        ).first()
        filas.append({"variable": c, "n": r["n"], "media": r["media"], "mediana": r["p"][1],
                      "desv_est": r["desv_est"], "min": r["min"], "p25": r["p"][0], "p75": r["p"][2],
                      "p95": r["p"][3], "max": r["max"], "asimetria": r["asimetria"]})
    return pd.DataFrame(filas).set_index("variable")


desc25 = resumen_num(df25, VARS_NUM)
desc25
""")

code(r"""
s = desc25.loc["salario_mensual"]
print(f"Salario: media Q{s.media:,.2f} vs mediana Q{s.mediana:,.2f} -> diferencia Q{s.media - s.mediana:,.2f} "
      f"({100 * (s.media / s.mediana - 1):.1f}% sobre la mediana); asimetría = {s.asimetria:.2f}")
print(f"El 5% mejor pagado gana más de Q{s.p95:,.2f}; el máximo es {s['max'] / s.mediana:,.1f} veces la mediana.")
""")

md("### 2.1 Distribución de registros entre categorías ocupacionales, niveles educativos y dominios")

code(r"""
def frecuencias(df, col, orden=None):
    pdf = df.groupBy(col).count().toPandas()
    pdf["pct"] = 100 * pdf["count"] / pdf["count"].sum()
    if orden:
        pdf[col] = pd.Categorical(pdf[col], categories=[o for o in orden if o in set(pdf[col])], ordered=True)
        return pdf.sort_values(col)
    return pdf.sort_values("count", ascending=False)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (col, orden) in zip(axes, [("categoria_ocupacional", None), ("nivel_educativo", ORDEN_EDU), ("dominio", None)]):
    f = frecuencias(df25, col, orden)
    sns.barplot(data=f, x="count", y=col, ax=ax, color="#4C72B0")
    for i, (n, p) in enumerate(zip(f["count"], f["pct"])):
        ax.text(n, i, f" {n:,} ({p:.1f}%)", va="center", fontsize=9)
    ax.set_title(f"Registros por {col.replace('_', ' ')}")
    ax.set_xlabel("Registros (2025)"); ax.set_ylabel("")
    ax.set_xlim(0, f["count"].max() * 1.35)
plt.tight_layout(); plt.show()
""")

md(r"""
**Interpretación.** *(Revisar con los valores obtenidos.)* La categoría **empleado de empresa privada** concentra la mayor parte de los asalariados, seguida por jornaleros/peones y empleados de gobierno; el servicio doméstico es el grupo más pequeño. En educación predominan los niveles de primaria y diversificado, y la educación superior representa una proporción menor. Los tres dominios tienen tamaños que dependen del diseño muestral (no de la población), lo que refuerza que los conteos no ponderados no son proporciones nacionales. Las categorías con pocos registros (p. ej., postgrado o `DESCONOCIDO`) producirán estimaciones más inestables.
""")

md("### 2.2 Forma de la distribución del salario")

code(r"""
n25 = df25.count()
muestra = (df25.sample(fraction=min(1.0, N_MUESTRA_GRAF / n25), seed=SEED)
           .limit(N_MUESTRA_GRAF).select(VARS_NUM + ["nivel_educativo", "categoria_ocupacional", "dominio"]).toPandas())
print(f"Muestra para gráficos: {len(muestra):,} de {n25:,} registros")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ax = axes[0]
sns.histplot(muestra["salario_mensual"], bins=80, ax=ax, color="#4C72B0")
ax.axvline(s.media, color="crimson", ls="--", label=f"Media (total) Q{s.media:,.0f}")
ax.axvline(s.mediana, color="black", ls="-", label=f"Mediana (total) Q{s.mediana:,.0f}")
ax.set_title("Salario mensual — escala lineal")
ax.set_xlabel("Salario mensual (Q)"); ax.legend()
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

ax = axes[1]
sns.histplot(muestra["salario_mensual"], bins=80, log_scale=True, ax=ax, color="#55A868")
ax.axvline(s.media, color="crimson", ls="--", label="Media")
ax.axvline(s.mediana, color="black", ls="-", label="Mediana")
ax.set_title("Salario mensual — ESCALA LOGARÍTMICA (eje X en log10)")
ax.set_xlabel("Salario mensual (Q, escala log)"); ax.legend()
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
plt.tight_layout(); plt.show()
""")

md(r"""
**¿El salario presenta una distribución simétrica o asimétrica?** Es **asimétrica positiva (sesgada a la derecha)**: el coeficiente de asimetría calculado en Spark sobre todos los registros es claramente mayor que 0, la mayoría de salarios se agrupa en valores bajos/medios y una cola larga de salarios altos se extiende a la derecha. En escala logarítmica la distribución se vuelve mucho más simétrica, lo que indica un comportamiento aproximadamente log‑normal. La escala log solo se usa para visualizar; el objetivo del modelado sigue siendo `salario_mensual` en quetzales.

**¿Qué diferencia existe entre su media y su mediana?** La **media es mayor que la mediana** (ver la diferencia impresa arriba). La media es sensible a los valores extremos de la cola derecha, que la "jalan" hacia arriba; la mediana representa mejor al asalariado típico. Por eso en las comparaciones entre grupos se usa la mediana. Los salarios extremos **no se eliminan**: son reales en la muestra, pero influirán fuertemente en métricas cuadráticas como el RMSE en la parte de modelado.
""")

md("### 2.3 Salario mediano por nivel educativo y categoría ocupacional")

code(r"""
def mediana_por(df, col, valor="salario_mensual"):
    return (df.groupBy(col).agg(F.count("*").alias("n"),
                                F.expr(f"percentile({valor}, 0.5)").alias("mediana"),
                                F.expr(f"percentile({valor}, 0.25)").alias("p25"),
                                F.expr(f"percentile({valor}, 0.75)").alias("p75"),
                                F.mean(valor).alias("media")).toPandas())


med_edu = mediana_por(df25, "nivel_educativo")
med_edu["nivel_educativo"] = pd.Categorical(med_edu["nivel_educativo"], [o for o in ORDEN_EDU if o in set(med_edu["nivel_educativo"])], ordered=True)
med_edu = med_edu.sort_values("nivel_educativo")
med_cat = mediana_por(df25, "categoria_ocupacional").sort_values("mediana")

fig, axes = plt.subplots(1, 2, figsize=(17, 5))
for ax, tabla, col in [(axes[0], med_edu, "nivel_educativo"), (axes[1], med_cat, "categoria_ocupacional")]:
    y = np.arange(len(tabla))
    ax.barh(y, tabla["mediana"], color="#4C72B0", label="Mediana")
    ax.errorbar(tabla["mediana"], y, xerr=[tabla["mediana"] - tabla["p25"], tabla["p75"] - tabla["mediana"]],
                fmt="none", ecolor="black", capsize=4, label="P25–P75")
    ax.set_yticks(y, [f"{c} (n={n:,})" for c, n in zip(tabla[col], tabla["n"])])
    for yi, m in zip(y, tabla["mediana"]):
        ax.text(m, yi + 0.3, f"Q{m:,.0f}", fontsize=9)
    ax.set_title(f"Salario mediano por {col.replace('_', ' ')} (2025)")
    ax.set_xlabel("Salario mensual (Q)")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="lower right")
plt.tight_layout(); plt.show()
display(med_edu.set_index("nivel_educativo"), med_cat.set_index("categoria_ocupacional"))
""")

code(r"""
fig, ax = plt.subplots(figsize=(14, 5))
orden_m = [o for o in ORDEN_EDU if o in set(muestra["nivel_educativo"])]
sns.boxplot(data=muestra, x="nivel_educativo", y="salario_mensual", hue="categoria_ocupacional",
            order=orden_m, showfliers=False, ax=ax)
ax.set_yscale("log")
ax.set_title("Salario por nivel educativo y categoría (muestra; ESCALA LOGARÍTMICA; sin atípicos dibujados)")
ax.set_ylabel("Salario mensual (Q, log)"); ax.set_xlabel("")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
ax.legend(title="", fontsize=8, loc="upper left")
plt.tight_layout(); plt.show()
""")

md(r"""
**Interpretación.** *(Revisar con los valores obtenidos.)* El salario mediano **aumenta con el nivel educativo**; el salto es más pronunciado a partir de diversificado y, sobre todo, en educación superior y postgrado, donde también se amplía el rango intercuartílico. Por categoría ocupacional, los **empleados de gobierno** muestran la mediana más alta, seguidos por empleados de empresa privada; **jornaleros/peones** y **servicio doméstico** tienen las medianas más bajas. Parte de la diferencia entre categorías se relaciona con la composición educativa de cada grupo (el boxplot conjunto muestra que dentro de un mismo nivel también hay brechas entre categorías). Son asociaciones descriptivas, no efectos causales.
""")

md("### 2.4 Tamaño de la muestra analítica y salario mediano por trimestre")

code(r"""
por_trim = (df25.groupBy("periodo_archivo")
            .agg(F.count("*").alias("n"), F.expr("percentile(salario_mensual, 0.5)").alias("mediana"),
                 F.mean("salario_mensual").alias("media"))
            .orderBy("periodo_archivo").toPandas())
por_trim = por_trim.merge(cascada.loc["0_registros_originales", PERIODOS_2025].rename("originales"),
                          left_on="periodo_archivo", right_index=True)
por_trim["pct_retenido"] = 100 * por_trim["n"] / por_trim["originales"]

fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax1.bar(por_trim["periodo_archivo"], por_trim["n"], color="#A1C9F4", label="Registros analíticos")
for x, n, p in zip(por_trim["periodo_archivo"], por_trim["n"], por_trim["pct_retenido"]):
    ax1.text(x, n, f"{n:,}\n({p:.1f}% del archivo)", ha="center", va="bottom", fontsize=9)
ax1.set_ylabel("Registros analíticos"); ax1.set_ylim(0, por_trim["n"].max() * 1.3)
ax2 = ax1.twinx()
ax2.plot(por_trim["periodo_archivo"], por_trim["mediana"], color="crimson", marker="o", label="Salario mediano")
ax2.plot(por_trim["periodo_archivo"], por_trim["media"], color="gray", marker="s", ls="--", label="Salario medio")
ax2.set_ylabel("Salario (Q)"); ax2.grid(False)
ax2.set_ylim(0, por_trim["media"].max() * 1.25)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left")
ax1.set_title("Muestra analítica y salario por trimestre de 2025")
plt.tight_layout(); plt.show()
por_trim
""")

md(r"""
**Interpretación.** *(Revisar con los valores obtenidos.)* El tamaño de la muestra analítica es similar entre trimestres (la proporción retenida de cada archivo es estable), por lo que ningún trimestre domina el conjunto de entrenamiento. El salario mediano varía poco entre trimestres; las diferencias pequeñas pueden deberse a la rotación de la muestra y a la estacionalidad (p. ej., cosechas que aumentan el peso de jornaleros en ciertos trimestres), no necesariamente a cambios reales en los salarios. La media es más volátil que la mediana porque depende de unos pocos salarios extremos.
""")

# ---------------------------------------------------------------------------
md(r"""
---
# 3. Relaciones entre variables numéricas

Correlación de **Pearson** con `VectorAssembler` + `Correlation.corr()` sobre **todos** los registros elegibles de 2025.
""")

code(r"""
va_corr = VectorAssembler(inputCols=VARS_NUM, outputCol="vec_corr")
vec = va_corr.transform(df25).select("vec_corr")
pearson = Correlation.corr(vec, "vec_corr", "pearson").head()[0].toArray()
spearman = Correlation.corr(vec, "vec_corr", "spearman").head()[0].toArray()   # complemento: robusta a asimetría

etiquetas = ["Salario", "Edad", "Antigüedad", "Horas"]
corr_p = pd.DataFrame(pearson, index=etiquetas, columns=etiquetas)
corr_s = pd.DataFrame(spearman, index=etiquetas, columns=etiquetas)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, m, t in [(axes[0], corr_p, "Pearson (principal)"), (axes[1], corr_s, "Spearman (complementaria)")]:
    sns.heatmap(m, annot=True, fmt=".3f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Coeficiente"})
    ax.set_title(f"Correlación {t} — 2025 (n={n25:,})")
plt.tight_layout(); plt.show()
corr_p
""")

code(r"""
cs = corr_p["Salario"].drop("Salario").sort_values(key=np.abs, ascending=False)
print("Correlación de Pearson con el salario (ordenada por magnitud):")
print(cs.round(3).to_string())
print(f"\nEdad vs antigüedad: Pearson = {corr_p.loc['Edad', 'Antigüedad']:.3f}, Spearman = {corr_s.loc['Edad', 'Antigüedad']:.3f}")
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
for ax, x in zip(axes, ["edad", "antiguedad", "horas_semanales"]):
    ax.scatter(muestra[x], muestra["salario_mensual"], s=6, alpha=0.25)
    ax.set_yscale("log")
    ax.set_xlabel(x); ax.set_ylabel("Salario (Q, ESCALA LOG)")
    ax.set_title(f"Salario vs {x} (muestra)")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(muestra["edad"], muestra["antiguedad"], s=6, alpha=0.25)
ax.plot([15, muestra["edad"].max()], [0, muestra["edad"].max() - 15], color="crimson", ls="--", label="antigüedad = edad − 15")
ax.set_xlabel("Edad"); ax.set_ylabel("Antigüedad (años)"); ax.legend()
ax.set_title("Edad vs antigüedad (muestra)")
plt.tight_layout(); plt.show()
""")

md(r"""
**¿Qué variables presentan mayor asociación lineal con el salario?** *(Revisar con los valores obtenidos.)* Las correlaciones de Pearson con el salario son **positivas pero débiles** en todas las variables numéricas. La **antigüedad** suele ser la de mayor asociación (más años en el empleo se asocian a mejores salarios), seguida por la **edad**; las **horas habituales** tienen una asociación lineal muy baja, porque muchas jornadas largas corresponden a ocupaciones de bajo salario (jornaleros, servicio doméstico). Que ninguna correlación sea alta indica que el salario depende en gran medida de factores categóricos (educación, categoría ocupacional, dominio) y de relaciones no lineales. Además, Pearson es sensible a la cola de salarios extremos: la correlación de Spearman (por rangos) sirve para comprobar si la relación monótona es más fuerte que la lineal.

**¿Existe relación entre edad y antigüedad?** Sí: es la correlación **más alta** de la matriz y es positiva. Tiene sentido estructural, porque la antigüedad está acotada por la edad (nadie puede tener más antigüedad que años de vida laboral); por eso el diagrama muestra un "triángulo": las personas jóvenes solo pueden tener poca antigüedad, mientras que las mayores pueden tener mucha o poca (cambios de empleo). Esta colinealidad moderada debe considerarse al interpretar coeficientes de la regresión lineal y al elegir variables para el clustering.
""")

# ---------------------------------------------------------------------------
md(r"""
---
# 4. Segmentación de perfiles mediante KMeans

### 4.1 Selección de variables — ¿incluir el salario?

Se comparan dos especificaciones:

| Opción | Variables | Justificación |
|---|---|---|
| **A (principal)** | edad, antigüedad, horas semanales | Perfiles según características personales/laborales observadas, sin usar el objetivo |
| **B (comparación)** | edad, antigüedad, horas semanales, log10(salario) | Evalúa qué ocurre si se incluye el salario |

Argumentos para **no** incluir el salario en la segmentación principal:
1. Es la **variable objetivo** del modelado supervisado; construir perfiles con ella y luego describir el salario por perfil sería circular.
2. Su **fuerte asimetría** haría que, sin transformar, unos pocos salarios extremos definieran clusters casi individuales; aun en log domina la distancia.
3. La pregunta de negocio es *qué perfiles de trabajadores existen*, y luego *cómo difiere el salario entre ellos*: el salario se usa para **describir** los clusters, no para formarlos.

Las variables se **estandarizan** (`StandardScaler`, media 0 y desviación 1) porque KMeans usa distancia euclidiana y, sin escalar, las variables con mayor rango (antigüedad en años vs horas) dominarían.

Las variables categóricas no se incluyen directamente en KMeans (las distancias euclidianas sobre variables *one‑hot* no son apropiadas); se usan después para caracterizar los grupos.
""")

code(r"""
VARS_A = ["edad", "antiguedad", "horas_semanales"]
VARS_B = VARS_A + ["log_salario"]
K_VALORES = [2, 3, 4, 5]

base_km = df25.withColumn("log_salario", F.log10("salario_mensual")).cache()
evaluador = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette",
                                distanceMeasure="squaredEuclidean")


def pipeline_kmeans(cols, k):
    return Pipeline(stages=[
        VectorAssembler(inputCols=cols, outputCol="x"),
        StandardScaler(inputCol="x", outputCol="features", withMean=True, withStd=True),
        KMeans(k=k, seed=SEED, featuresCol="features", predictionCol="cluster", maxIter=100, initSteps=5),
    ])


resultados, modelos = [], {}
for opcion, cols in [("A: sin salario", VARS_A), ("B: con log(salario)", VARS_B)]:
    for k in K_VALORES:
        modelo = pipeline_kmeans(cols, k).fit(base_km)
        pred = modelo.transform(base_km)
        tam = pred.groupBy("cluster").count().toPandas()["count"]
        resultados.append({
            "opcion": opcion, "k": k,
            "silueta": evaluador.evaluate(pred),
            "wssse": modelo.stages[-1].summary.trainingCost,
            "pct_cluster_mas_pequeno": 100 * tam.min() / tam.sum(),
        })
        modelos[(opcion, k)] = modelo

res_km = pd.DataFrame(resultados)
res_km
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for opcion, g in res_km.groupby("opcion"):
    axes[0].plot(g["k"], g["silueta"], marker="o", label=opcion)
    axes[1].plot(g["k"], g["wssse"], marker="o", label=opcion)
axes[0].set_title("Coeficiente de silueta (mayor es mejor)"); axes[0].set_xlabel("K"); axes[0].set_ylabel("Silueta")
axes[1].set_title("WSSSE — método del codo"); axes[1].set_xlabel("K"); axes[1].set_ylabel("Suma de cuadrados intra‑cluster")
for ax in axes:
    ax.set_xticks(K_VALORES); ax.legend()
axes[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
plt.tight_layout(); plt.show()
""")

md(r"""
### 4.2 Criterio del grupo para elegir K

Se elige K en la **opción A** con el siguiente criterio:
1. **Mayor coeficiente de silueta** (cohesión y separación de los grupos, calculado sobre todos los registros);
2. siempre que el cluster más pequeño contenga **al menos el 5 %** de los registros (grupos útiles y estables);
3. contrastado con el **codo** del WSSSE y con que los perfiles resultantes sean **interpretables**.
""")

code(r"""
cand = res_km[(res_km["opcion"] == "A: sin salario") & (res_km["pct_cluster_mas_pequeno"] >= 5)]
if cand.empty:
    cand = res_km[res_km["opcion"] == "A: sin salario"]
K_ELEGIDO = int(cand.sort_values("silueta", ascending=False).iloc[0]["k"])
print(f"K elegido (opción A): {K_ELEGIDO}")
print(res_km[res_km["k"] == K_ELEGIDO].set_index("opcion")[["silueta", "pct_cluster_mas_pequeno"]])

modelo_km = modelos[("A: sin salario", K_ELEGIDO)]
seg = modelo_km.transform(base_km).drop("x", "features").cache()
""")

md(r"""
**¿Vale la pena incluir el salario?** Compare las siluetas de ambas opciones en la tabla y el gráfico anteriores. Al agregar `log_salario` se añade una dimensión más y la silueta suele **disminuir** (o no mejorar significativamente) porque el salario no forma grupos bien separados: varía de forma continua dentro de cada combinación de edad, antigüedad y jornada. Además, los clusters de la opción B tienden a partirse simplemente por "salario alto / bajo", lo que no aporta un perfil nuevo y sería circular respecto al modelo supervisado. Por estas razones, **la segmentación final no incluye el salario** y este se usa para *describir* los perfiles.
""")

md("### 4.3 Perfil de cada cluster")

code(r"""
perfil = (seg.groupBy("cluster").agg(
    F.count("*").alias("n"),
    F.round(F.mean("edad"), 1).alias("edad_media"),
    F.expr("percentile(edad, 0.5)").alias("edad_mediana"),
    F.round(F.mean("antiguedad"), 1).alias("antig_media"),
    F.expr("percentile(antiguedad, 0.5)").alias("antig_mediana"),
    F.round(F.mean("horas_semanales"), 1).alias("horas_media"),
    F.expr("percentile(horas_semanales, 0.5)").alias("horas_mediana"),
    F.expr("percentile(salario_mensual, 0.5)").alias("salario_mediano"),
    F.round(F.mean("salario_mensual"), 0).alias("salario_medio"),
).orderBy("cluster").toPandas().set_index("cluster"))
perfil.insert(1, "pct", 100 * perfil["n"] / perfil["n"].sum())

# Centroides en unidades originales (desestandarizados)
scaler_m = modelo_km.stages[1]
centros = np.array(modelo_km.stages[-1].clusterCenters()) * scaler_m.std.toArray() + scaler_m.mean.toArray()
centros = pd.DataFrame(centros, columns=VARS_A).round(1).rename_axis("cluster")
display(perfil, centros)
""")

code(r"""
def composicion(col, orden=None):
    t = seg.groupBy("cluster", col).count().toPandas().pivot(index="cluster", columns=col, values="count").fillna(0)
    t = 100 * t.div(t.sum(axis=1), axis=0)
    if orden:
        t = t[[o for o in orden if o in t.columns]]
    return t


comp_cat = composicion("categoria_ocupacional")
comp_edu = composicion("nivel_educativo", ORDEN_EDU)
comp_dom = composicion("dominio")

fig, axes = plt.subplots(1, 3, figsize=(20, 1.2 + 0.8 * K_ELEGIDO + 2), gridspec_kw={"width_ratios": [4, 7, 3]})
for ax, t, titulo in [(axes[0], comp_cat, "Categoría ocupacional"), (axes[1], comp_edu, "Nivel educativo"), (axes[2], comp_dom, "Dominio")]:
    sns.heatmap(t, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100, cbar=False, ax=ax)
    ax.set_title(f"% de cada cluster por {titulo.lower()}"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    plt.setp(ax.get_xticklabels(), ha="right")
plt.tight_layout(); plt.show()
""")

code(r"""
muestra_seg = (seg.sample(fraction=min(1.0, N_MUESTRA_GRAF / n25), seed=SEED).limit(N_MUESTRA_GRAF)
               .select(VARS_A + ["salario_mensual", "cluster"]).toPandas())
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, v in zip(axes, VARS_A + ["salario_mensual"]):
    sns.boxplot(data=muestra_seg, x="cluster", y=v, ax=ax, showfliers=False, color="#A1C9F4")
    ax.set_title(v + (" (ESCALA LOG)" if v == "salario_mensual" else ""))
    if v == "salario_mensual":
        ax.set_yscale("log"); ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
plt.suptitle(f"Distribución de variables por cluster (muestra de {len(muestra_seg):,}; salario no usado para formar los clusters)")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(muestra_seg["edad"], muestra_seg["antiguedad"], c=muestra_seg["cluster"], s=np.clip(muestra_seg["horas_semanales"] / 6, 1, 30),
                cmap="tab10", alpha=0.5)
ax.legend(*sc.legend_elements(), title="Cluster")
ax.set_xlabel("Edad"); ax.set_ylabel("Antigüedad (años)"); ax.set_title("Clusters en edad vs antigüedad (tamaño ∝ horas)")
plt.tight_layout(); plt.show()
""")

md(r"""
### 4.4 Descripción de los clusters

Para que la etiqueta de cada grupo se base en los resultados, se compara el centroide de cada cluster con los terciles globales de edad y antigüedad y con umbrales de jornada (< 40 h parcial, 40–48 h completa, > 48 h extendida), y se añade la categoría ocupacional y el nivel educativo predominantes.
""")

code(r"""
terc = {c: df25.select(F.expr(f"percentile({c}, array(0.3333, 0.6667))")).first()[0] for c in ["edad", "antiguedad"]}


def nivel(v, cortes, nombres):
    return nombres[0] if v < cortes[0] else (nombres[1] if v < cortes[1] else nombres[2])


def describir(c):
    e = nivel(centros.loc[c, "edad"], terc["edad"], ["jóvenes", "adultos", "adultos mayores"])
    a = nivel(centros.loc[c, "antiguedad"], terc["antiguedad"], ["baja antigüedad", "antigüedad media", "alta antigüedad"])
    h = centros.loc[c, "horas_semanales"]
    j = "jornada parcial" if h < 40 else ("jornada completa" if h <= 48 else "jornada extendida")
    cat = comp_cat.loc[c].idxmax(); edu = comp_edu.loc[c].idxmax()
    return f"{e.capitalize()}, {a}, {j}; predomina {cat.lower()} ({comp_cat.loc[c].max():.0f}%) y nivel {edu.lower()}"


perfil["descripcion"] = [describir(c) for c in perfil.index]
pd.set_option("display.max_colwidth", 200)
perfil[["n", "pct", "edad_mediana", "antig_mediana", "horas_mediana", "salario_mediano", "descripcion"]]
""")

md(r"""
**Interpretación de la segmentación.** *(Revisar y ajustar con los perfiles obtenidos.)*

- El K elegido maximiza la silueta respetando un tamaño mínimo de grupo, por lo que ofrece la partición más clara sin crear segmentos marginales.
- Los clusters se diferencian principalmente por **etapa de vida laboral** (edad y antigüedad, que están correlacionadas) y por **intensidad de la jornada**. Típicamente aparecen: (i) trabajadores **jóvenes con poca antigüedad**; (ii) trabajadores **adultos con alta antigüedad**, con mayor presencia de empleados de gobierno y niveles educativos más altos, y con el **salario mediano más alto**; y (iii) un grupo con **jornadas extendidas** o, en el otro extremo, **jornadas parciales**, donde pesan más jornaleros y servicio doméstico.
- Aunque el salario no se usó para formar los grupos, su mediana difiere entre clusters, lo que muestra que el perfil laboral está asociado con el ingreso. Sin embargo, la dispersión salarial dentro de cada cluster es amplia, por lo que el perfil por sí solo no basta para predecir el salario: esto motiva los modelos supervisados de la segunda parte, que añaden educación, categoría ocupacional y dominio.
- Los resultados describen los **registros analizados sin ponderar**; con `FACTOR` se podría estimar el tamaño poblacional de cada perfil.
""")

code(r"""
seg.select("periodo_archivo", "NUM_HOGAR", "NUM_PERSONA", "cluster").write.mode("overwrite") \
   .parquet(str(DATA_PROC / "clusters_2025.parquet"))
print("Etiquetas de cluster guardadas (solo para análisis descriptivo; no se usan como predictor).")
""")

# ---------------------------------------------------------------------------
nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "Laboratorio7_Spark.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook escrito en", out)
