from Server.HttpServer.ClientServer import client_server

client_server.run("127.0.0.1", 8377, debug=True, use_reloader=False)
# client_server.run("101.133.214.221", 8377, debug=True, use_reloader=False)

