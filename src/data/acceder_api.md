Primero de todo te tienes que instalar el packete de Kaggle:

pip install kaggle

Despues tienes que escribir estos comandos:

# 1. Configuramos las credenciales (Corregidas)
$env:KAGGLE_USERNAME="TU_USER_NAME"
$env:KAGGLE_KEY="TU_API_KEY"

# 2. Descargamos el dataset específico (fíjate que usamos 'datasets' no 'competitions')
kaggle datasets download -d yasserh/instacart-online-grocery-basket-analysis-dataset --unzip

Y ya lo tendras, para poder comprobar si tienes bien la api puedes hacer ejecutar este comando:

kaggle datasets list