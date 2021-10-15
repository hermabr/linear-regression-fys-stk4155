from data_generation import FrankeData
from linear_regression_models import LinearRegression
from sklearn.linear_model import Lasso as LassoSKLearn


class Lasso(LinearRegression):
    def __init__(self, degree, lambda_):
        """Constructor for the lasso regression model

        Parameters
        ----------
            degree : int
                The degree for the lasso model
            lambda_ : float
                The value of the penalty term
        """
        super().__init__(degree)
        self.model = LassoSKLearn(lambda_)

    def fit(self, x, y, z):
        """Fits the data using sklearn lasso

        Parameters
        ----------
            x : np.array
                The x values for which to fit the model
            y : np.array
                The y values for which to fit the model
            z : np.array
                The z values for which to fit the model
        """
        X = self.generate_design_matrix(x, y)
        self.model.fit(X, z)

    def predict(self, x, y):
        """Custom predict method using the sklearn

        Parameters
        ----------
            x : np.array
                The x values for which to predict
            y : np.array
                The y values for which to predict

        Returns
        -------
            z_tilde : np.array
                The predicted z-tilde-values
        """
        X = self.generate_design_matrix(x, y)
        z_tilde = self.model.predict(X)
        return z_tilde

    def __repr__(self):
        return "Lasso"
