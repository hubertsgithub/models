#based on: https://gist.github.com/yukixz/5835965
#!/usr/bin/python

import http.server

import random
import html
import io
import os
import socketserver
import sys
import urllib.parse
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Directory containing pairs of images and visualized predictions")
parser.add_argument("--port", default=8000, type=int, help="port to serve on")
args = parser.parse_args()

SUFFIX = urllib.parse.quote('.xhtml')

DEEPLAB_DATASET_PATH = '/mnt/data/projects-hubert/tensorflow/models/research/deeplab/datasets/'
assert os.path.exists(DEEPLAB_DATASET_PATH)

legend_path = os.path.join(DEEPLAB_DATASET_PATH, 'legend.png')
assert os.path.exists(legend_path)

print ('#############################################################')
print ('COPYING LEGEND FROM {} TO {}'.format(legend_path, args.dir))
print ('#############################################################')
shutil.copy(legend_path, os.path.join(args.dir, 'zzzlegend.png'))

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith(SUFFIX):
            path = self.path.replace(SUFFIX, '')
            f = self.send_file_in_html(path)
        else:
            f = self.send_head()

        if f:
            self.copyfile(f, self.wfile)
            f.close()

    def send_file_in_html(self, path):
        enc = sys.getfilesystemencoding()
        path = self.translate_path(path)
        (dirname, filename) = os.path.split(path)

        try:
            list = os.listdir(dirname)
            list.sort(key=lambda a: a.lower())
        except os.error:
            list = []

        if list.index(filename) % 2 != 0:
            filename = list[list.index(filename) - 1]
        try:
            if list.index(filename) >= 2:
                prevname = list[list.index(filename)-2] + SUFFIX
            else:
                prevname = filename + SUFFIX
            filename_pred = list[list.index(filename)+1]
            nextname = list[list.index(filename)+2] + SUFFIX
            randname = random.sample(list, 1)[0] + SUFFIX
        except ValueError:
            self.send_error(404, "File not found")
            return None
        except IndexError:
            nextname = ''

        r=[]
        r.append('<html>')
        r.append('<head><meta http-equiv="Content-Type" content="text/html; charset=%s"></head>' % enc)
        r.append('<br>')
        r.append('<a href="./{}"><button type="button">PREVIOUS</button></a>'.format(prevname))
        r.append('<a href="./{}"><button type="button">NEXT</button></a>'.format(nextname))
        r.append('<br>')
        r.append('<a href="./"><button type="button">home</button></a>')
        r.append('<br>')
        r.append('<a href="./{}"><button type="button">random</button></a>'.format(randname))
        r.append('<br>')
        r.append('<body><a href="%s"><img src="%s"></img></a></body>'\
            % (os.path.join('./',nextname), os.path.join('./',filename) ))
        r.append('<body><a href="%s"><img src="%s"></img></a></body>'\
            % (os.path.join('./',nextname), os.path.join('./',filename_pred) ))
        r.append('<body><img src="%s" height="350"></a></body>'\
            % ('zzzlegend.png'))
            #% ('legend.png'))
        r.append('</html>')


        encoded = '\n'.join(r).encode(enc)
        f = io.BytesIO()
        f.write(encoded)
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=%s" % enc)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        return f

    def list_directory(self, path):
        ''' Overwriting SimpleHTTPRequestHandler.list_directory()
            Modify marked with `####`
        '''
        try:
            list = os.listdir(path)
        except os.error:
            self.send_error(404, "No permission to list directory")
            return None
        list.sort(key=lambda a: a.lower())
        r = []
        displaypath = html.escape(urllib.parse.unquote(self.path))
        enc = sys.getfilesystemencoding()
        title = 'Directory listing for %s' % displaypath
        r.append('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" '
                 '"http://www.w3.org/TR/html4/strict.dtd">')
        r.append('<html>\n<head>')
        r.append('<meta http-equiv="Content-Type" '
                 'content="text/html; charset=%s">' % enc)
        r.append('<title>%s</title>\n</head>' % title)
        r.append('<body>\n<h1>%s</h1>' % title)
        r.append('<hr>\n<ul>')
        for name in list:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            # Append / for directories or @ for symbolic links
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"
            if os.path.islink(fullname):
                displayname = name + "@"
                # Note: a link to a directory displays with @ and links with /
            if os.path.isfile(fullname):        ####
                linkname = name + SUFFIX       ####
            r.append('<li><a href="%s">%s</a></li>'
                    % (urllib.parse.quote(linkname), html.escape(displayname)))
        r.append('</ul>\n<hr>\n</body>\n</html>\n')
        encoded = '\n'.join(r).encode(enc)
        f = io.BytesIO()
        f.write(encoded)
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=%s" % enc)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        return f


if __name__=='__main__':
    os.chdir(args.dir)

    socketserver.TCPServer.allow_reuse_address = True

    httpd = socketserver.TCPServer(("", args.port), Handler)
    print("Serving on 0.0.0.0:{} ...".format(args.port))

    httpd.serve_forever()
