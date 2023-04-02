import pymysql
import json

if __name__ == '__main__':
    # 连接数据库
    conn = pymysql.connect(host='127.0.0.1', user='root', passwd='123456', port=3306, db='pest', charset='utf8')
    cur = conn.cursor()  # 生成游标对象
    sql = "select * from `mypest` "  # SQL语句
    cur.execute(sql)  # 执行SQL语句
    data = cur.fetchall()  # 通过fetchall方法获得数据
    for i in range(len(data)):
        to_json = {'no': data[i][0],
                   'name': data[i][1].replace('\t', '') if data[i][1] is not None else data[i][1],
                   'info': data[i][2].replace('\t', '') if data[i][2] is not None else data[i][2],
                   'danger': data[i][3].replace('\t', '')if data[i][3] is not None else data[i][3],
                   'morph': data[i][4].replace('\t', '')if data[i][4] is not None else data[i][4],
                   'gen': data[i][5].replace('\t', '')if data[i][5] is not None else data[i][5],
                   'habit': data[i][6].replace('\t', '')if data[i][6] is not None else data[i][6],
                   'measure': data[i][7].replace('\t', '')if data[i][7] is not None else data[i][7]}
        with open(f'./data/{i}.json', 'w') as f:
            need = json.dumps(to_json, ensure_ascii=False)
            print(need)
            f.write(need)
