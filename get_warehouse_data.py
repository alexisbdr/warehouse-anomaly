import sys
import logging
import os
from datetime import datetime
from enum import Enum
from itertools import product, chain
import threading
import concurrent.futures

sys.path.append("../onetrack_neural_net_infrastructure")

from data_tools.image_request import ImageRequest, ImageFilter
from data_tools.image_data import ImageData

from dynamo_query import query_global_secondary_index

def curr_nano():
    return int(time.time() * 1000000000)

class WillmerConfig(Enum):
    TEST_FOLDER = "warehouse_dataset/DSCWillmer/Test"
    TRAIN_FOLDER = "warehouse_dataset/DSCWillmer/Train"
    DOWNLOAD_FOLDER = "warehouse_dataset/DSCWillmer"


    DAYS = []
    MONTHS = [1]
    YEARS = [2020]

    EQUIPMENT_TYPE = 'forklift'
    CUSTOMER = 'DSC Logistics'
    PLANT = 'DSC Wilmer'

    MACHINE_IDS = ['Crown-601-1A560570','Crown-101-1A561553','Crown-107-1A561554','Crown-109-1A561640','Crown-113-1A561851','Crown-202-1A562058','Crown-102-1A561639','Crown-110-1A561641','Crown-124-1A562061','Crown-120-1A561859','Crown-117-1A561647','Crown-135-1A561849','Crown-116-1A561853','Crown-203-1A562059','Crown-32-XXXX','Crown-105-1A561644','Crown-5-XXXX','Crown-125-1A561855','Crown-123-1A561857','Crown-126-1A560920','Crown-112-1A560826','Crown-114-1A561556','Crown-8-XXXX','Crown-118-1A562062','Crown-128-1A560919','Crown-106-1A560921','Crown-201-1A562063','Crown-104-1A561642','Crown-115-1A561854','Crown-103-1A562064','Crown-28-1A561850','Crown-100-1A562060','Crown-204-1A562065','Crown-200-1A562057','Crown-121-1A561858','Crown-119-1A561555','Crown-108-1A561646','Crown-111-1A561643','Crown-11-XXXX','Crown-22-XXXX','Crown-14-XXXX','Crown-36-XXXX','Crown-24-1A561849','Crown-12-XXXX','Crown-1-XXXX','Crown-130-1A561850','Crown-23-XXXX','Crown-31-XXXX','Crown-19-XXXX','Crown-17-XXXX','Crown-13-XXXX','Crown-129-1A561852','Crown-33-XXXX','Crown-122-1A561856','Crown-30-XXXX','Crown-29-XXXX','Crown-20-XXXX','Crown-6-XXXX']


    DEVICES = ['ifm-ips-213',
        'ifm-ips-231', 'ifm-ips-233', 'ifm-ips-214', 'ifm-ips-222', 'ifm-ips-220', 'ifm-ips-238', 'ifm-ips-225', 'ifm-ips-212', 'ifm-ips-223', 'ifm-ips-244', 'ifm-ips-216', 'ifm-ips-246', 'ifm-ips-211', 'ifm-ips-241', 'ifm-ips-232', 'ifm-ips-217', 'ifm-ips-227', 'ifm-ips-224', 'ifm-ips-240', 'ifm-ips-218', 'ifm-ips-243', 'ifm-ips-242', 'ifm-ips-229', 'ifm-ips-247', 'ifm-ips-215', 'ifm-ips-245', 'ifm-ips-226', 'ifm-ips-221', 'ifm-ips-210', 'ifm-ips-239', 'ifm-ips-230']


class FairbournConfig(Enum):
    TEST_FOLDER = "warehouse_dataset/DSCFairbourn/Test"
    TRAIN_FOLDER = "warehouse_dataset/DSCFairbourn/Train"
    DOWNLOAD_FOLDER = "warehouse_dataset/DSCFairbourn"

    DAYS = []
    MONTHS = []
    YEARS = []

    EQUIPMENT_TYPE = 'forklift'
    CUSTOMER = 'DSC Logistics'
    PLANT = 'DSC Fairbourn'


def downloadImagesByPlant(plantConfig):
    """
    Uses the query_global_second_index method to query by machine_id and timestam
    """

    for machine, year, month in product(
        plantConfig.MACHINE_IDS.value,
        plantConfig.YEARS.value,
        plantConfig.MONTHS.value,
    ):
        print(f"starting request for date: {year}-{month} and customer {plantConfig.CUSTOMER.value} at plant {plantConfig.PLANT.value} on machine {machine}")

        first_ts_hour = int(
            datetime.strptime(
                f"{year}-{month}-15 00:00:00",
                "%Y-%m-%d %H:%M:%S")
            .timestamp()
            * 1000000000)
        last_ts_hour = int(
            datetime.strptime(
                f"{year}-{month}-31 11:59:59",
                "%Y-%m-%d %H:%M:%S")
            .timestamp()
            * 1000000000)

        print(f"searching for images between: {first_ts_hour} and {last_ts_hour}")

        resp = query_global_secondary_index(machine, last_ts_hour, first_ts_hour)
        print(list(enumerate(resp)))


        def download_thread(obj):
            count, _images = obj
            print(f"starting thread {count}")
            img_data = ImageData.from_dict(_images)
            image_timestamp = img_data.metadata.timestamp_utc_ns
            #Check if device folder exists
            folder_loc = os.path.join(plantConfig.DOWNLOAD_FOLDER.value, machine, str(month), "15-31")
            if not os.path.exists(folder_loc):
                os.makedirs(folder_loc)
            num = str(count).zfill(4)
            day = str(img_data.metadata.day)
            hour = str(img_data.metadata.hour)
            date = f"{year}-{month}-{day}-{hour}"
            time_num = int(image_timestamp / 10000000)
            file_loc = f"{folder_loc}/img{num}_{date}_{time_num}.jpeg"
            img_data.download_image(file_loc=file_loc)
            print(f"Dowloaded image {file_loc}")

            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(download_thread, enumerate(resp))

if __name__ == "__main__":
    downloadImagesByPlant(WillmerConfig)

