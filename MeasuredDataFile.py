from nptdms import TdmsFile
import pandas as pd

class MeaseredDataFile():
    def __init__(self, file_path, file_type, config):
        self.config = config
        self.file_type = file_type
        self.file_path = file_path.replace("?", self.file_type)
        self.column_names = self.config.get(self.file_type, "column_names").split(",")
        self.use_column = list(map(bool, self.config.get(self.file_type, "use_column").replace("False", "").split(",")))
        self.convert_constants = [float(item) if item != "None" else None for item in self.config.get(self.file_type, "convert_constants").split(",")]
        self.frequency = int(self.config.get(self.file_type, "frequency"))
        self.data_raw = self.load_file()
        self.data_converted = pd.DataFrame()

    def load_file(self):
        df_temp = TdmsFile(self.file_path).as_dataframe()
        df_temp.columns = self.column_names

        for i, const in enumerate(self.convert_constants):
            name = self.column_names[i]
            # if self.use_column[i] and const and self.column_names[i] == "pos_ref":
            #     df_temp[self.column_names[i]] = (df_temp[self.column_names[i]]*55.14706-23.71324)/1000
            if self.use_column[i] and const:
                df_temp[self.column_names[i]] *= const

        for i, name in enumerate(self.column_names):
            if not self.use_column[i]:
                df_temp.pop(name)

        return df_temp