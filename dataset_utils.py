import os
from datetime import datetime

def filename_to_date(filename):
    basename = os.path.basename(filename)
    year = int(basename[17:21])
    month = int(basename[22:24])
    day = int(basename[25:27])
    hour = int(basename[28:30])
    minute = int(basename[30:32])
    second = int(basename[32:34])
    date = datetime(year, month, day, hour, minute, second)

    return date


def date_to_filename(date, wavelength, extension, level="lev1", duration="12s"):
    """
    datetimeオブジェクトから指定の形式のファイル名を生成する

    Parameters:
    - date: datetimeオブジェクト
    - level: レベル指定 (例: "lev1")
    - duration: 観測の期間 (例: "12s")
    - wavelength: 波長 (例: "211")

    Returns:
    - filename: 生成されたファイル名
    """

    filename = f"aia.{level}_euv_{duration}.{date.strftime('%Y-%m-%dT%H%M%SZ')}.{wavelength}.image_{level}.{extension}"
    return filename
