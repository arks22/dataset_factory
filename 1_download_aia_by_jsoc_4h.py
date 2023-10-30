import argparse
import astropy.units as u
from sunpy.net import jsoc
from sunpy.net import attrs as a
import datetime
import calendar

parser = argparse.ArgumentParser()
parser.add_argument('dst_dir', type=str)
parser.add_argument('--wave', type=int, required=True)
parser.add_argument('--delta', type=int, required=True)
parser.add_argument('--start_year', type=int, default=2010)
parser.add_argument('--end_year', type=int, default=2024)
parser.add_argument('--start_month', type=int, default=1)
parser.add_argument('--end_month', type=int, default=13)
args = parser.parse_args()


# (1) FITSをダウンロードして一つのディレクトリに格納
# 4時間ごとのサンプリング幅で1日くらい

series = 'aia.lev1_euv_12s'
email  = 'sasaki.10222022@gmail.com'
#email  = 'f22c017d@mail.cc.niigata-u.ac.jp'

client = jsoc.JSOCClient()  

years  = list(range(args.start_year, args.end_year))
months = list(range(args.start_month, args.end_month))


for year in years:
    for month in months:
        last_day_of_month = calendar.monthrange(year, month)[1]

        start_date = '{}/{}/1T00:00:00'.format(year,month)
        end_date = '{}/{}/{}T23:59:59'.format(year,month,last_day_of_month) 

        response = client.search(a.Time(start_date,end_date),
                             a.jsoc.Series(series),
                             a.jsoc.Segment('image'),
                             a.Wavelength(args.wave*u.AA),
                             a.Sample(args.delta*u.hour), 
                             a.jsoc.Notify(email))
        print(response)
        requests = client.fetch(response, path=args.dst_dir, progress=True, sleep=45, overwrite=True)
