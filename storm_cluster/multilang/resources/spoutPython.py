import storm
import time

#Verander pad
database_location = "C:/Users/Arthu/Desktop/thesis_realtime_ids/csecicids2018-clean/"

DATA_0302 = database_location + "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
DATA_0301 = database_location + "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
DATA_0228 = database_location + "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0223 = database_location + "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0222 = database_location + "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0221 = database_location + "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0220 = database_location + "Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0216 = database_location + "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0215 = database_location + "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"
DATA_0214 = database_location + "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"

files = [DATA_0214, DATA_0215, DATA_0216, DATA_0220, DATA_0221, DATA_0222, DATA_0223, DATA_0228, DATA_0301, DATA_0302]


class SpoutPython(storm.Spout):
    def initialize(self, conf, context):
        self._conf = conf
        self._context = context
        self.tuple_id = 0
        self.file_nr = 0

        storm.logInfo("Spout starting...")
        try:  
            self.f = open(DATA_0302, 'r')
            self.f.readline() #skip eerste lijn van dataset
            time.sleep(60)
            storm.logInfo("Spout ready...")
        except OSError:
            storm.logInfo("File error")
            exit()
        
    def activate(self):
        pass

    def deactivate(self):
        pass

    def ack(self, id):
        pass

    def fail(self, id):
        pass

    def nextTuple(self):
        try:
            line = self.f.readline()
            storm.emit([line], id=self.tuple_id)
            self.tuple_id += 1

        except EOFError:
            exit()
            
SpoutPython().run()