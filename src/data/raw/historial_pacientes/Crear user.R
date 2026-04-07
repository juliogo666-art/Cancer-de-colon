# --- CONFIGURACIÓN ---
ruta_entrada <- "cancer_risk_clean.csv"
ruta_salida <- "cancer_risk_final.csv"

# --- LISTAS DE NOMBRES ---

nombres_h <- c(
  "Juan","Carlos","Pedro","David","Luis","Javier","Antonio","Jose","Manuel","Francisco",
  "Ricardo","Alejandro","Roberto","Fernando","Daniel","Jorge","Raul","Sergio","Adrian","Ivan",
  "Miguel","Angel","Jose Antonio","Jose Luis","Juan Carlos","Vicente","Ramon","Joaquin","Oscar","Ruben",
  "Marc","Jordi","Pau","Joan","Oriol","Albert","Xavier","Pol","Eric","Bernat",
  "Hugo","Marcos","Pablo","Diego","Alvaro","Mario","Victor","Ignacio","Gonzalo","Guillermo"
)

nombres_m <- c(
  "Maria","Lucia","Ana","Laura","Carmen","Elena","Marta","Paula","Isabel","Cristina",
  "Sara","Julia","Claudia","Alba","Andrea","Nuria","Montserrat","Silvia","Monica","Beatriz",
  "Patricia","Raquel","Irene","Natalia","Helena","Marina","Esther","Rosa","Teresa","Dolores",
  "Laia","Mireia","Neus","Meritxell","Berta","Carla","Ariadna","Aina","Sonia","Gemma",
  "Sofia","Martina","Daniela","Valeria","Noelia","Ines","Miriam","Lorena","Victoria","Angela"
)

apellidos <- c(
  "Garcia","Martinez","Lopez","Sanchez","Perez","Gomez","Rodriguez","Fernandez","Moreno","Jimenez",
  "Pascual","Ruiz","Alonso","Vidales","Gutierrez","Navarro","Torres","Dominguez","Vazquez","Ramos",
  "Gil","Ramirez","Serrano","Blanco","Molina","Morales","Suarez","Ortega","Delgado","Castro",
  "Ortiz","Rubio","Marin","Sanz","Nuñez","Iglesias","Medina","Cortes","Castillo","Santos",
  "Ferrer","Soler","Puig","Vila","Serra","Marti","Vidal","Roca","Riera","Font",
  "Planellas","Dalmau","Gisbert","Bernat","Cano","Mendez","Cruz","Prieto","Flores","Cabrera"
)

# --- FUNCIONES ---

generar_dni_unicos <- function(n) {
  letras <- c("T","R","W","A","G","M","Y","F","P","D","X","B","N","J","Z","S","Q","V","H","L","C","K","E")
  numeros <- sample(10000000:99999999, n, replace = FALSE)
  letra <- letras[(numeros %% 23) + 1]
  paste0(numeros, letra)
}

generar_nuss_bcn_unicos <- function(n) {
  provincia <- "08"
  base_nums <- sample(10000000:99999999, n, replace = FALSE)
  
  resultados <- sapply(base_nums, function(num) {
    numero_base <- paste0(provincia, sprintf("%08d", num))
    control <- sprintf("%02d", as.numeric(numero_base) %% 97)
    paste0(numero_base, control)
  })
  
  return(resultados)
}

# --- LEER CSV ORIGINAL ---
datos_originales <- read.csv(ruta_entrada, stringsAsFactors = FALSE)
n_registros <- nrow(datos_originales)

# --- CREAR DATASET NUEVO ---
nuevos_datos <- data.frame(
  Patient_ID = datos_originales$Patient_ID,
  stringsAsFactors = FALSE
)

# --- GENERAR DATOS ---
generos <- sample(c("Male", "Female"), n_registros, replace = TRUE)

nuevos_datos$Gender <- generos
nuevos_datos$City <- "Barcelona"
nuevos_datos$Country <- "Spain"

nuevos_datos$nombre <- ifelse(
  nuevos_datos$Gender == "Male",
  sample(nombres_h, n_registros, replace = TRUE),
  sample(nombres_m, n_registros, replace = TRUE)
)

nuevos_datos$apellido1 <- sample(apellidos, n_registros, replace = TRUE)
nuevos_datos$apellido2 <- sample(apellidos, n_registros, replace = TRUE)

nuevos_datos$Age <- sample(18:90, n_registros, replace = TRUE)

nuevos_datos$altura_cm <- ifelse(
  nuevos_datos$Gender == "Male",
  round(rnorm(n_registros, 175, 7), 1),
  round(rnorm(n_registros, 162, 6), 1)
)

imc_aleatorio <- runif(n_registros, 20, 32)
nuevos_datos$peso_kg <- round(imc_aleatorio * (nuevos_datos$altura_cm/100)^2, 1)

nuevos_datos$dni <- generar_dni_unicos(n_registros)
nuevos_datos$nuss <- generar_nuss_bcn_unicos(n_registros)

# --- EVITAR ERRORES DE FORMATO EN EXCEL ---
nuevos_datos$dni <- as.character(nuevos_datos$dni)
nuevos_datos$nuss <- as.character(nuevos_datos$nuss)

# --- GUARDAR ---
write.csv(nuevos_datos, ruta_salida, row.names = FALSE, quote = TRUE)

print(paste("✅ CSV generado correctamente con", n_registros, "filas:", ruta_salida))

