# Leer CSV
ruta <- "datos_combinados_global_extendido_3.csv"
datos <- read.csv(ruta, stringsAsFactors = FALSE)


n <- nrow(datos)

# Listas de nombres
nombres_h <- c("Juan","Carlos","Pedro","David","Luis","Javier")
nombres_m <- c("Maria","Lucia","Ana","Laura","Carmen","Elena")
apellidos <- c("Garcia","Martinez","Lopez","Sanchez","Perez","Gomez")

# Nombre según Gender existente
datos$nombre <- ifelse(
  datos$Gender == "Male",
  sample(nombres_h, n, replace = TRUE),
  sample(nombres_m, n, replace = TRUE)
)

# Apellidos
datos$apellido1 <- sample(apellidos, n, replace = TRUE)
datos$apellido2 <- sample(apellidos, n, replace = TRUE)

# Altura según género (más realista)
datos$altura_cm <- ifelse(
  datos$Gender == "Male",
  round(rnorm(n, 175, 20), 1),
  round(rnorm(n, 165, 15), 1)
)

# Peso correlacionado con altura (IMC aprox 22–30)
imc <- runif(n, 22, 30)
datos$peso_kg <- round(imc * (datos$altura_cm/100)^2, 1)

datos$peso_kg <- ifelse(
  datos$Gender == "Male",
  round(rnorm(n, 75, 20), 1),
  round(rnorm(n, 65, 15), 1)
)


# DNI válido
generar_dni <- function(n){
  letras <- c("T","R","W","A","G","M","Y","F","P","D","X","B","N","J",
              "Z","S","Q","V","H","L","C","K","E")
  numeros <- sample(10000000:99999999, n)
  letra <- letras[(numeros %% 23) + 1]
  paste0(numeros, letra)
}

# 7. FUNCIÓN: Generar NUSS (Seguridad Social) para Barcelona (08)
generar_nuss_bcn <- function(n) {
  provincia <- "08" # Prefijo de Barcelona
  
  # Generamos un número secuencial de 8 dígitos
  # Usamos strings para evitar problemas de precisión con números muy largos
  base_num <- sample(10000000:99999999, n, replace = TRUE)
  
  # Cálculo del dígito de control (estándar SS España: número total mod 97)
  # Concatenamos provincia y base para el cálculo
  full_num_str <- paste0(provincia, base_num)
  
  # En R, para números tan grandes usamos as.numeric o transformamos
  control <- as.numeric(full_num_str) %% 97
  control_str <- sprintf("%02d", control) # Asegura 2 dígitos (ej: 05)
  
  paste0(provincia, base_num, control_str)
}

datos$dni <- generar_dni(n)
datos$nuss <- generar_nuss_bcn(n)

# Guardar CSV actualizado
write.csv(datos, ruta, row.names = FALSE)