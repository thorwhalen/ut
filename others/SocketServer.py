__author__ = 'thor'

import socket

#create an INET, STREAMing socket
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#bind the socket to a public host,
# and a well-known port
# serversocket.bind((socket.gethostname(), 3743))
serversocket.bind(('localhost', 3743))
#become a server socket
serversocket.listen(5)

while True:
    #accept connections from outside
    (connection, address) = serversocket.accept()
    buf = connection.recv(64)
    if len(buf) > 0:
        print(buf)
        break

serversocket.close()


### THEN DO THIS IN ANOTHER SCRIPT OR IPYTHON TO TEST IT:
# import socket
# clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# clientsocket.connect(('localhost', 3743))
# clientsocket.send('look at me! I passed through a socket!')

