import numpy as np
from imageio import imread
from sklearn.model_selection import train_test_split
from config import DEFAULT_NOISE_LEVEL


class Data:
    def __init__(self):
        """Empty initialized for the abstract data class"""
        pass

    def scale_data(self, data):
        """Scales the data by scaling to values from 0 to 1, then subtracting the mean

        Parameters
        ----------
            data : np.array
                The data for which to scale

        Returns
        -------
            data : np.array
                A scaled version of the data
        """
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data -= np.mean(data)
        return data

    def store_data(self, x, y, z, test_size):
        """Stores the data, either as only x, y, and z, or splitting the x, y, and z in train/test and saving all

        Parameters
        ----------
            x : np.array
                The x data to save
            y : np.array
                The y data to save
            z : np.array
                The z data to save
            test_size : float/None
                The test size for which to store the data. None means no test data
        """
        x, y, z = np.ravel(x), np.ravel(y), np.ravel(z)

        if not test_size:
            self._x = x
            self._y = y
            self._z = z
        else:
            (
                self._x_train,
                self._x_test,
                self._y_train,
                self._y_test,
                self._z_train,
                self._z_test,
            ) = train_test_split(x, y, z, test_size=test_size)

    def train_test_split(self, test_size):
        """Splits the data into a train and test data by using sklearn train_test_split

        Parameters
        ----------
            test_size : float
                The size of the test data, compared to the train data
        """
        self.check_property("_x")
        self.check_property("_y")
        self.check_property("_z")
        (
            self._x_train,
            self._x_test,
            self._y_train,
            self._y_test,
            self._z_train,
            self._z_test,
        ) = train_test_split(self.x, self.y, self.z, test_size=test_size)

    def check_property(self, name):
        """Check if a property with a given name is present

        Parameters
        ----------
            name : str
                The name of the property for which to check

        Returns
        -------
            attribute :
                The attribute for the given name

        Raises
        ------
            AttributeError :
                Raises an attribute error if the given attribute does not exist
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(
                f"The franke data does not have the attribute '{name[1:]}'. You can only access 'x', 'y', 'z' if there is no test split, and 'x_train', 'y_train', 'z_train', 'x_test', 'y_test' and 'z_test' if there is a test split"
            )

    @property
    def x(self):
        """Get the x-value if it exists

        Returns
        -------
            x : np.array
                Returns the attribute x if it exists
        """
        return self.check_property("_x")

    @property
    def y(self):
        """Get the y-value if it exists

        Returns
        -------
            y : np.array
                Returns the attribute y if it exists
        """
        return self.check_property("_y")

    @property
    def z(self):
        """Get the z-value if it exists

        Returns
        -------
            z : np.array
                Returns the attribute z if it exists
        """
        return self.check_property("_z")

    @property
    def x_train(self):
        """Get the x_train-value if it exists

        Returns
        -------
            x_train : np.array
                Returns the attribute x_train if it exists
        """
        return self.check_property("_x_train")

    @property
    def y_train(self):
        """Get the y_train-value if it exists

        Returns
        -------
            y_train : np.array
                Returns the attribute y_train if it exists
        """
        return self.check_property("_y_train")

    @property
    def z_train(self):
        """Get the z_train-value if it exists

        Returns
        -------
            z_train : np.array
                Returns the attribute z_train if it exists
        """
        return self.check_property("_z_train")

    @property
    def x_test(self):
        """Get the x_test-value if it exists

        Returns
        -------
            x_test : np.array
                Returns the attribute x_test if it exists
        """
        return self.check_property("_x_test")

    @property
    def y_test(self):
        """Get the y_test-value if it exists

        Returns
        -------
            y_test : np.array
                Returns the attribute y_test if it exists
        """
        return self.check_property("_y_test")

    @property
    def z_test(self):
        """Get the z_test-value if it exists

        Returns
        -------
            z_test : np.array
                Returns the attribute z_test if it exists
        """
        return self.check_property("_z_test")

    def scale_down_data_size(self, x, y, z, scale):
        """Scale down the data size by slicing out only some x and y indices

        Parameters
        ----------
            x : np.array
                The x values for which to scale down
            y : np.array
                The y values for which to scale down
            z : np.array
                The z values for which to scale down
            scale : float
                The scale for which to scale down the data

        Returns
        -------
            x : np.array
                The sliced down version of the x-values
            y : np.array
                The sliced down version of the y-values
            z : np.array
                The sliced down version of the z-values
        """
        x_indices = np.random.choice(len(x), size=int(len(x) * scale), replace=False)
        y_indices = np.random.choice(len(y), size=int(len(y) * scale), replace=False)

        x_indices = np.sort(x_indices)
        y_indices = np.sort(y_indices)

        x = x[x_indices]
        y = y[y_indices]

        x_indices, y_indices = np.meshgrid(x_indices, y_indices)
        z = z[y_indices, x_indices]

        self.dimensions = z.shape

        return x, y, z


class FrankeData(Data):
    def __init__(
        self,
        N,
        random_noise=True,
        random_positions=True,
        scale_data=True,
        test_size=None,
        noise_level=DEFAULT_NOISE_LEVEL,
    ):
        """The data class for the franke data

        Parameters
        ----------
            N : int
                The number of elements in the x and y directions
            random_noise : bool
                Adds random noise if true
            random_positions : bool
                Sets random positions if true, else uses linspace to generate evenly spaced values
            scale_data : bool
                A bool specifying if the data should be scaled
            test_size : float/None
                Uses a specified size (0 to 1) as the test data
            noise_level : float
                The sigma value for the noise level
        """
        super().__init__()

        self.dimensions = (N, N)

        if random_positions:
            data = np.random.rand(N * 2).reshape(N, 2)
            x = np.sort(data[:, 0])
            y = np.sort(data[:, 1])
        else:
            x = np.linspace(0, 1, N)
            y = np.linspace(0, 1, N)

        x, y = np.meshgrid(x, y)

        if random_noise:
            z = self.NoisyFrankeFunction(x, y, noise_level)
        else:
            z = self.FrankeFunction(x, y)

        if scale_data:
            z = self.scale_data(z)

        self.store_data(x, y, z, test_size)

    @staticmethod
    def FrankeFunction(x, y):
        """The franke function written as a numpy expression

        Parameters
        ----------
            x : np.array
                The x-values for which to generate z-values from the franke function
            y : np.array
                The y-values for which to generate z-values from the franke function

        Returns
        -------
            z : np.array
                The z values from the franke function
        """
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4

    @staticmethod
    def NoisyFrankeFunction(x, y, noise_level):
        """A noisy version of the franke function

        Parameters
        ----------
            x : np.array
                The x-values for which to generate z-values from the franke function
            y : np.array
                The y-values for which to generate z-values from the franke function
            noise_level : float
                The sigma value for the amount of noise to add to the fnction

        Returns
        -------
            z : np.array
                The z values from the franke function with added noise
        """
        noise = FrankeData.generate_noise(x.shape, noise_level=noise_level)
        return FrankeData.FrankeFunction(x, y) + noise

    @staticmethod
    def generate_noise(N, noise_level=DEFAULT_NOISE_LEVEL):
        """Generates noise from a normal distribution

        Parameters
        ----------
            N : int
                The number of noises to add
            noise_level : float
                The sigma value for the amount of noise to add to the fnction

        Returns
        -------
            noise : np.array
                An array containing N values of noise with sigma noise_level
        """
        noise = np.random.normal(loc=0.0, scale=noise_level, size=N)
        return noise


class TerrainData(Data):
    def __init__(self, filename, test_size=None, scale_data=True, scale_data_size=0.2):
        """The data class for the terrain function

        Parameters
        ----------
            filename : str
                The filename for which to read the terrain data
            test_size : float/None
                Uses a specified size (0 to 1) as the test data
            scale_data : bool
                Specifies if the data should be scaled
            scale_data_size : float
                The amount to downscale the data size

        """
        super().__init__()

        terrain = np.array(imread(filename))

        self.dimensions = terrain.shape

        x = np.linspace(0, 1, terrain.shape[1])
        y = np.linspace(0, 1, terrain.shape[0])

        if scale_data_size:
            x, y, terrain = self.scale_down_data_size(x, y, terrain, scale_data_size)

        x, y = np.meshgrid(x, y)

        x, y, z = np.ravel(x), np.ravel(y), np.ravel(terrain)

        if scale_data:
            z = self.scale_data(z)

        self.store_data(x, y, z, test_size)
