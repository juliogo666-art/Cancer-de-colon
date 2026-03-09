set.seed(123)

n <- 10

nombres <- c("Juan","María","Carlos","Lucía","Pedro","Ana","David","Laura")
apellidos <- c("García","Martínez","López","Sánchez","Pérez","Gómez")

datos <- data.frame(
  nombre = sample(nombres, n, replace = TRUE),
  apellido1 = sample(apellidos, n, replace = TRUE),
  apellido2 = sample(apellidos, n, replace = TRUE),
  edad = sample(18:80, n, replace = TRUE),
  altura_cm = round(rnorm(n, 170, 10), 1),
  peso_kg = round(rnorm(n, 70, 15), 1)
)


letras <- c("T","R","W","A","G","M","Y","F","P","D","X","B","N","J",
            "Z","S","Q","V","H","L","C","K","E")

generar_dni <- function(n){
  numeros <- sample(10000000:99999999, n)
  letra <- letras[(numeros %% 23) + 1]
  paste0(numeros, letra)
}

datos$dni <- generar_dni(n)


datos