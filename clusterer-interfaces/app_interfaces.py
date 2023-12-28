import collections.abc
collections.MutableMapping = collections.abc.MutableMapping

import os
from flask_cors import CORS
from dotenv import load_dotenv
from flask import Flask, request
from werkzeug.datastructures import FileStorage
from flask_restplus import Api, Resource, fields
from utils import remote_clustering, response_func
from waitress import serve

load_dotenv(".env")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

CORS(app, resources={r"/api/*": {"enabled": "true",
                                 "headers": [
                                     "Authorization",
                                     "Origin",
                                     "Access-Control-Request-Method",
                                     "Access-Control-Request-Headers",
                                     "Accept",
                                     "Accept-Charset",
                                     "Accept-Encoding",
                                     "Accept-Language",
                                     "Cache-Control",
                                     "Content-Type",
                                     "Cookie",
                                     "DNT",
                                     "Pragma",
                                     "Referer",
                                     "User-Agent",
                                     "X-Forwarded-For"
                                 ]
                                 }}, supports_credentials=True)

name_space = api.namespace('api', 'На вход поступает JSON, возвращает *.xlsx')

input_data = name_space.model("Insert JSON",
                              {"texts": fields.List(fields.String(description="Insert texts")),
                               "scores_list": fields.List(fields.Float(description="Distance", required=True))}, )


CLUSTERING_URL = os.environ.get("CLUSTERING_URL")
if CLUSTERING_URL is None: raise Exception('Env var CLUSTERING_URL not defined')


@name_space.route('/json_excel')
class ClusteringJsonExcel(Resource):
    @name_space.expect(input_data)
    def post(self):
        """POST method on input json file with texts and score, output clustering texts as xlsx file."""
        json_data = request.json
        clustering_texts_df = remote_clustering(json_data, CLUSTERING_URL, upload_type="json")
        return response_func(clustering_texts_df)


"""================== csv files ==================="""
def api_configurator(name_space):
    """"""
    upload_parser = name_space.parser()
    upload_parser.add_argument("file", type=FileStorage, location='files', required=True,
                               help="тексты должны быть в первом столбце")
    upload_parser.add_argument("scores_list", type=float, action='append', required=True,
                               help="кластеризация для каждого значения скора")
    return upload_parser


# https://debugthis.dev/flask/2019-10-09-building-a-restful-api-using-flask-restplus/
csv_name_space = api.namespace('api', 'загрузка файла csv с текстами (текст для кластеризации в первом столбце)'
                                      ' и score (число)')
csv_upload_parser = api_configurator(csv_name_space)


@csv_name_space.route('/csv_csv')
@csv_name_space.expect(csv_upload_parser)
class ClusteringCsvCsv(Resource):
    def post(self):
        """POST method on input csv file with texts and score, output clustering texts  as csv file."""
        args = csv_upload_parser.parse_args()
        clustering_texts_df = remote_clustering(args, CLUSTERING_URL, upload_type="csv")
        return response_func(clustering_texts_df, response_type="csv")


"""================== excel files ==================="""
# https://debugthis.dev/flask/2019-10-09-building-a-restful-api-using-flask-restplus/
excel_name_space = api.namespace('api', 'загрузка файла excel с текстами (текст для кластеризации в первом столбце)'
                                        ' и score (число)')
excel_upload_parser = api_configurator(excel_name_space)


@excel_name_space.route('/excel_excel')
@excel_name_space.expect(excel_upload_parser)
class ClusteringEcxelExcel(Resource):
    def post(self):
        """POST method on input xlsx file with texts and score, output clustering texts  as xlsx file."""
        args = excel_upload_parser.parse_args()
        clustering_texts_df = remote_clustering(args, CLUSTERING_URL)
        return response_func(clustering_texts_df)


if __name__ == "__main__":
    # serve(app, host="0.0.0.0", port=8080)
    app.run(host='0.0.0.0', port=8080)
