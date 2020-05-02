import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv("./data/whr2017.csv")

    x = dataset[['gdp',
                 'family',
                 'lifexp',
                 'freedom',
                 'corruption',
                 'generosity',
                 'dystopia'
                 ]]

    y = dataset[['score']]

    # Separacion de los datos
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # Modelo de regresion lineal
    modal_linear = LinearRegression().fit(x_train, y_train)
    y_prediction_linear = modal_linear.predict(x_test)

    # Modelo Lasso
    model_lasso = Lasso(alpha=0.02).fit(x_train, y_train)
    y_prediction_lasso = model_lasso.predict(x_test)

    # Modelo Ridge
    model_ridge = Ridge(alpha=1).fit(x_train, y_train)
    y_prediction_ridge = model_ridge.predict(x_test)

    # Modelo Elastic Net
    model_elastic = ElasticNet(random_state=0, alpha=1).fit(x_train, y_train)
    y_prediction_elastic = model_elastic.predict(x_test)

    # Perdida lineal
    linear_loss = mean_squared_error(y_test, y_prediction_linear)
    print("Linear: ", linear_loss)

    # Perdida Lasso
    lasso_loss = mean_squared_error(y_test, y_prediction_lasso)
    print("Lasso: ", lasso_loss)

    # Perdida Ridge
    ridge_loss = mean_squared_error(y_test, y_prediction_ridge)
    print("Ridge: ", ridge_loss)

    # Perdida Elastic Net
    elastic_loss = mean_squared_error(y_test, y_prediction_elastic)
    print("Elastic Net: ", elastic_loss)

    # Imprimiendo los coeficientes
    print("=" * 32)
    print("Coef LASSO")
    print(model_lasso.coef_)

    print("=" * 32)
    print("Coef RIDGE")
    print(model_ridge.coef_)

    print("=" * 32)
    print("Coef ELASTIC")
    print(model_elastic.coef_)
