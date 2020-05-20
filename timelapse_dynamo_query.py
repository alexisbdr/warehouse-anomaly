
from boto3.dynamodb.conditions import Key, Attr
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('device_data_requests')
s3 = boto3.client('s3')

def scan_table_allpages(table, filtering_exp):
    response = table.scan(FilterExpression=filtering_exp)
    items = response['Items']
    while True:
        if response.get('LastEvaluatedKey'):

            response = table.scan(FilterExpression=filtering_exp, ExclusiveStartKey=response['LastEvaluatedKey'])
            items += response['Items']
        else:
            break
        return items

def query_by_plant(plant: str):
    search_attr = Attr('status').eq('uploading') | Attr('status').eq('complete')
    response = scan_table_allpages(table, search_attr)
    return resp

for resp in response:
    s3.download_file(resp["target_output_bucket"], resp["target_output_key"], os.path.join("data", resp["target_output_key"].split("/")[-1]))
