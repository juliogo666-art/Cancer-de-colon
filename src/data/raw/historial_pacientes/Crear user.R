# --- CONFIGURACIÓN ---
n_registros <- 5000
ruta_salida <- "nuevos_pacientes_5000.csv"

# Nombres Masculinos (50 opciones)
nombres_h <- c(
  "Juan", "Carlos", "Pedro", "David", "Luis", "Javier", "Antonio", "Jose", "Manuel", "Francisco",
  "Ricardo", "Alejandro", "Roberto", "Fernando", "Daniel", "Jorge", "Raul", "Sergio", "Adrian", "Ivan",
  "Miguel", "Angel", "Jose Antonio", "Jose Luis", "Juan Carlos", "Vicente", "Ramon", "Joaquin", "Oscar", "Ruben",
  "Marc", "Jordi", "Pau", "Joan", "Oriol", "Albert", "Xavier", "Pol", "Eric", "Bernat",
  "Hugo", "Marcos", "Pablo", "Diego", "Alvaro", "Mario", "Victor", "Ignacio", "Gonzalo", "Guillermo"
)

# Nombres Femeninos (50 opciones)
nombres_m <- c(
  "Maria", "Lucia", "Ana", "Laura", "Carmen", "Elena", "Marta", "Paula", "Isabel", "Cristina",
  "Sara", "Julia", "Claudia", "Alba", "Andrea", "Nuria", "Montserrat", "Silvia", "Monica", "Beatriz",
  "Patricia", "Raquel", "Irene", "Natalia", "Helena", "Marina", "Esther", "Rosa", "Teresa", "Dolores",
  "Laia", "Mireia", "Neus", "Meritxell", "Berta", "Carla", "Ariadna", "Aina", "Sonia", "Gemma",
  "Sofia", "Martina", "Daniela", "Valeria", "Noelia", "Ines", "Miriam", "Lorena", "Victoria", "Angela"
)

# Apellidos (60 opciones)
# Incluye los más comunes de España y algunos típicos de la zona de Barcelona
apellidos <- c(
  "Garcia", "Martinez", "Lopez", "Sanchez", "Perez", "Gomez", "Rodriguez", "Fernandez", "Moreno", "Jimenez",
  "Pascual", "Ruiz", "Alonso", "Vidales", "Gutierrez", "Navarro", "Torres", "Dominguez", "Vazquez", "Ramos",
  "Gil", "Ramirez", "Serrano", "Blanco", "Molina", "Morales", "Suarez", "Ortega", "Delgado", "Castro",
  "Ortiz", "Rubio", "Marin", "Sanz", "Nuñez", "Iglesias", "Medina", "Cortes", "Castillo", "Santos",
  "Ferrer", "Soler", "Puig", "Vila", "Serra", "Marti", "Vidal", "Roca", "Riera", "Font",
  "Planellas", "Dalmau", "Gisbert", "Bernat", "Cano", "Mendez", "Cruz", "Prieto", "Flores", "Cabrera"
)

# --- FUNCIONES DE GENERACIÓN ÚNICA ---

# Generar DNI único
generar_dni_unicos <- function(n) {
  letras <- c("T","R","W","A","G","M","Y","F","P","D","X","B","N","J","Z","S","Q","V","H","L","C","K","E")
  # Generamos números únicos usando sample sin reemplazo
  numeros <- sample(10000000:99999999, n, replace = FALSE)
  letra <- letras[(numeros %% 23) + 1]
  return(paste0(numeros, letra))
}

# Generar NUSS único (Barcelona 08)
generar_nuss_bcn_unicos <- function(n) {
  provincia <- "08"
  # Generar números base únicos
  base_nums <- sample(10000000:99999999, n, replace = FALSE)
  
  resultados <- sapply(base_nums, function(num) {
    full_str <- paste0(provincia, sprintf("%08d", num))
    control <- as.numeric(full_str) %% 97
    return(paste0(provincia, sprintf("%08d", num), sprintf("%02d", control)))
  })
  return(resultados)
}

# --- CREACIÓN DEL DATASET ---

# 1. Crear estructura base
generos <- sample(c("Male", "Female"), n_registros, replace = TRUE)

nuevos_datos <- data.frame(
  Patient_ID = 00000:(00000 + n_registros - 1), # IDs correlativos no repetidos
  Gender = generos,
  City = "Barcelona",   # Fijo para todos
  Country = "Spain",    # Fijo para todos
  stringsAsFactors = FALSE
)

# 2. Generar Nombres según Género
nuevos_datos$nombre <- ifelse(
  nuevos_datos$Gender == "Male",
  sample(nombres_h, n_registros, replace = TRUE),
  sample(nombres_m, n_registros, replace = TRUE)
)

# 3. Apellidos aleatorios
nuevos_datos$apellido1 <- sample(apellidos, n_registros, replace = TRUE)
nuevos_datos$apellido2 <- sample(apellidos, n_registros, replace = TRUE)

# 4. Edad aleatoria (ejemplo de 18 a 90 años)
nuevos_datos$Age <- sample(18:90, n_registros, replace = TRUE)

# 5. Altura y Peso realistas
nuevos_datos$altura_cm <- ifelse(
  nuevos_datos$Gender == "Male",
  round(rnorm(n_registros, 175, 7), 1), # Media 175, SD 7
  round(rnorm(n_registros, 162, 6), 1)  # Media 162, SD 6
)

# Cálculo de peso basado en IMC aleatorio entre 20 y 32
imc_aleatorio <- runif(n_registros, 20, 32)
nuevos_datos$peso_kg <- round(imc_aleatorio * (nuevos_datos$altura_cm/100)^2, 1)

# 7. Identificadores ÚNICOS
nuevos_datos$dni <- generar_dni_unicos(n_registros)
nuevos_datos$nuss <- generar_nuss_bcn_unicos(n_registros)

# --- GUARDAR ---
write.csv(nuevos_datos, ruta_salida, row.names = FALSE)

print(paste("✅ Proceso finalizado. Se han creado", n_registros, "registros en", ruta_salida))