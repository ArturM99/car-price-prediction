from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
from datetime import datetime
import pandas as pd
import dill

sched = BlockingScheduler(timezone=tzlocal.get_localzone_name())

df = pd.read_csv('Model/data/homework.csv')

with open('Model/cars_pipe.pkl', 'rb') as file:
    model = dill.load(file)

@sched.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(frac=0.05)
    data['preds'] = model['model'].predict(data)
    print(data[['id', 'preds']])

if __name__ == '__main__':
    sched.start()