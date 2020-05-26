import sys
import logging
from typing import (
    Mapping,
    Optional,
)
import boto3
from boto3.dynamodb.conditions import (
    ConditionBase,
    Key,
)

#Explicit Keys
table = boto3.resource( 'dynamodb',
                       region_name='us-east-1').Table("onetrack_neural_net_assets")

INDEX = "machine_id-timestamp_utc_ns-index"

def query_global_secondary_index(machine_id: str,
                                 timestamp_lt: int,
                                 timestamp_gt: int):
    key_condition = (Key("machine_id").eq(machine_id)
                     & Key("timestamp_utc_ns").between(timestamp_gt, timestamp_lt))
    return _query(key_condition)

def _query(key_condition: ConditionBase):
    records = []
    last_eval_key = {}
    while last_eval_key is not None:
        resp = _query_next_segment(key_condition,
                                   last_eval_key={})
        records.extend(resp["Items"])
        logging.info("Scanned count: %d",
                     resp["ScannedCount"])
        logging.info("Consumed capacity: %s",
                     resp["ConsumedCapacity"])
        last_eval_key = resp.get("last_eval_key", None)
        logging.info("Found %d matching records",
                     len(records))
        return records

def _query_next_segment(key_condition: ConditionBase,
                        last_eval_key: Optional[Mapping]) -> Mapping:
    params = {
        "IndexName": INDEX,
        "KeyConditionExpression": key_condition,
        "ReturnConsumedCapacity": "TOTAL",

    }
    if last_eval_key:
        params["ExclusiveStartKey"] = last_eval_key
    resp = table.query(**params)

    return resp
