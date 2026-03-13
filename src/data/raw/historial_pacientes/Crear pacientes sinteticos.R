# Leer CSV
ruta <- "datos_finales_Kaggle.csv"
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

generar_nuss_bcn <- function(n) {
  provincia <- "08" 
  
  # 1. Generar el número base de 8 dígitos como texto
  # Usamos sprintf para asegurar que siempre tenga 8 caracteres (rellena con 0 si es necesario)
  base_num <- sprintf("%08d", sample(0:99999999, n, replace = TRUE))
  
  # 2. Cálculo del dígito de control siguiendo la norma técnica:
  # Si el número base es menor a 10.000.000, se suma la provincia al número base.
  # Pero la forma más robusta es: concatenar provincia + base y hacer mod 97.
  
  resultados <- sapply(1:n, function(i) {
    # Concatenamos para el cálculo
    full_str <- paste0(provincia, base_num[i])
    
    # Convertimos a numérico de alta precisión para el módulo
    # Si no quieres instalar librerías, usamos una alternativa simple:
    val_num <- as.numeric(full_str)
    control <- val_num %% 97
    
    control_str <- sprintf("%02d", control)
    
    # IMPORTANTE: Retornar como cadena de texto para no perder el 08 inicial
    return(paste0(provincia, base_num[i], control_str))
  })
  
  return(resultados)
}

datos$dni <- generar_dni(n)
datos$nuss <- generar_nuss_bcn(nrow(datos))

# Guardar CSV actualizado
write.csv(datos, ruta, row.names = FALSE)