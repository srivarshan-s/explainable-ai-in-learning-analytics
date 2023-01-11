import pickle
import numpy as np

class CustomModel:

    def __init__(self):
        self.attr_1_model = pickle.load(open("models/cog_model.pkl", "rb"))
        self.attr_2_model = pickle.load(open("models/eff_model.pkl", "rb"))
        self.attr_3_model = pickle.load(open("models/reas_model.pkl", "rb"))
        self.arg_model = pickle.load(open("models/qual_model.pkl", "rb"))

    def predict(self, array):
        attr_1 = self.attr_1_model.predict(array, verbose=0)
        attr_2 = self.attr_2_model.predict(array, verbose=0)
        attr_3 = self.attr_3_model.predict(array, verbose=0)
        attr_1 = self.__decode(attr_1)
        attr_2 = self.__decode(attr_2)
        attr_3 = self.__decode(attr_3)
        array = self.__transform(attr_1, attr_2, attr_3, array)
        pred = self.arg_model.predict(array)
        return pred

    def __decode(self, array):
        new_array = []
        label_map = {
            0: "1 (Low)",
            1: "2 (Average)",
            2: "3 (High)",
        }
        for ele in array:
            new_array.append(label_map[np.argmax(ele)])
        return np.array(new_array)

    def __transform(self, attr_1, attr_2, attr_3, array):
        attr_1 = self.__encode(attr_1)
        attr_2 = self.__encode(attr_2)
        attr_3 = self.__encode(attr_3)
        array_new = []
        for idx, ele in enumerate(array):
            temp = np.concatenate((attr_1[idx], attr_2[idx], attr_3[idx], ele))
            array_new.append(temp)
        array = np.array(array_new)
        return array

    def __encode(self, array):
        new_array = []
        label_map = {
            "1 (Low)": np.array([0, 0, 1]),
            "2 (Average)": np.array([0, 1, 0]),
            "3 (High)": np.array([1, 0, 0]),
        }
        for ele in array:
            new_array.append(label_map[ele])
        return np.array(new_array)
