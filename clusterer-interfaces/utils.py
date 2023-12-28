import io
import requests
import pandas as pd
from flask import Response
from werkzeug.datastructures import Headers


def dataframe_handle(df: pd.DataFrame):
    """"""
    df.drop([0], axis=0, inplace=True)
    df.rename(columns={df.columns[0]: 'origin_queries',
                       df.columns[1]: 'all_queries',
                       df.columns[2]: 'unique_users',
                       df.columns[3]: 'well_transition',
                       df.columns[4]: 'without_clicks',
                       df.columns[5]: 'super_search',
                       df.columns[6]: 'super_fast_answers',
                       df.columns[7]: 'super_all'}, inplace=True)
    df["super_digital"] = df['all_queries'] * df['super_all']
    return df


def clustering_func(clustering_url, json_data):
    """"""
    clustering_texts_response = requests.post(clustering_url, json=json_data)
    clustering_texts = clustering_texts_response.json()
    return pd.DataFrame(clustering_texts["texts_with_labels"],
                        columns=["label", "cluster_name", "texts", "cluster_size"])


def remote_clustering(args, clustering_url, upload_type="excel") -> [{}]:
    """"""
    clustering_dataframes = []
    if upload_type == "json":
        input_data = {"texts": args["texts"], "scores": args['scores_list']}
        for score in input_data["scores"]:
            json_data = {"texts": input_data["texts"], "score": score}
            clustering_dataframe = clustering_func(clustering_url, json_data)
            clustering_dataframes.append({"clustering_dataframe": clustering_dataframe, "score": score})
    else:
        if upload_type == "excel":
            df = pd.read_excel(args['file'], header=None)
        else:
            """The function expects csv type of upload data with comma separated data"""
            df = pd.read_csv(args['file'], header=None)
        if df.shape[1] == 8:
            df = dataframe_handle(df)
            for score in args['scores_list']:
                json_data = {"texts": [str(tx) for tx in list(df['origin_queries'])], "score": score}
                clustering_dataframe = clustering_func(clustering_url, json_data)
                clustering_dataframe_with_stat = pd.merge(df, clustering_dataframe, left_on="origin_queries",
                                                          right_on="texts")
                clustering_dataframe_with_stat.drop(
                    ["texts", "origin_queries", "cluster_name", "cluster_size", "super_all"],
                    axis=1, inplace=True)
                grp = clustering_dataframe_with_stat.groupby("label", as_index=False).sum()
                grp["super_percent"] = grp["super_digital"] / grp["all_queries"]
                result_df = pd.merge(grp, clustering_dataframe, on="label")
                clustering_dataframes.append({"clustering_dataframe": result_df, "score": score})
        else:
            for score in args['scores_list']:
                json_data = {"texts": list(set(df[0])), "score": score}
                clustering_dataframe = clustering_func(clustering_url, json_data)
                clustering_dataframes.append({"clustering_dataframe": clustering_dataframe, "score": score})
    return clustering_dataframes


def response_func(clustering_dataframes, response_type="excel"):
    """"""
    headers = Headers()
    if response_type == "excel":
        headers.add('Content-Disposition', 'attachment', filename="clustering_results.xlsx")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            for d in clustering_dataframes:
                sheet_data = d["clustering_dataframe"]
                sheet_name = str(d["score"])
                # sheet_data.to_excel(writer, sheet_name=sheet_name, encoding='cp1251', index=False)
                sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
        # writer.save()
        writer.close()
        return Response(buffer.getvalue(),
                        mimetype='application/vnd.ms-excel',
                        headers=headers)

    else:
        headers.add('Content-Disposition', 'attachment', filename="clustering_results.csv")
        result_df = pd.DataFrame()
        for d in clustering_dataframes:
            df = d["clustering_dataframe"]
            df["score"] = d["score"]
            result_df = pd.concat((result_df, df))
        return Response(result_df.to_csv(index=False),
                        mimetype="text/csv",
                        headers=headers)
