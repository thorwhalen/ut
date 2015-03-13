from datetime import datetime
import ConfigParser
import argparse
import envoy
import boto
import os
import sys
import shutil
from boto.s3.key import Key
from ut.others.mo2s3 import default_conf

cfg = ConfigParser.SafeConfigParser()
cfg_file = os.path.expanduser("~/.mo2s3.cfg")
if not os.path.isfile(cfg_file):
    with open(cfg_file, "w") as f:
        f.write(default_conf)

cfg.read(cfg_file)

parser = argparse.ArgumentParser(description='Backup and restore MongoDB with Amazon S3.')
parser.add_argument("action", help="configure/backup/restore FILENAME/delete FILENAME/list/drop", action="store")
parser.add_argument('--host', help='Optional MongoDB Host', default=cfg.get("mongodb", "host"))
parser.add_argument('-u', '--username', help='Optional MongoDB username', default=cfg.get("mongodb", "username"))
parser.add_argument('-p', '--password', help='Optional MongoDB password', default=cfg.get("mongodb", "password"))
parser.add_argument('-d', '--db', help='Optional MongoDB database', default="")
parser.add_argument('-c', '--collection', help='Optional MongoDB collection', default="")
parser.add_argument('-b', '--bucket', help='AWS S3 bucket', default=cfg.get("aws", "s3_bucket"))
parser.add_argument('-k', '--folder', help='Folder (i.e. key prefix)', default=cfg.get("aws", "folder"))

parser.add_argument('-f', '--filename', help='File to delete/restore', default="")
parser.add_argument('-a', '--access-key', help='AWS access key', default=cfg.get("aws", "access_key"))
parser.add_argument('-s', '--secret-key', help='AWS secret key', default=cfg.get("aws", "secret_key"))


def make_mongo_params(args):
    mongo_params = ''
    if args['host']:
        mongo_params += " --host " + args["host"]
    if args["username"]:
        mongo_params += " --username " + args["username"] + " --password " + args["password"]
    if args["db"]:
        mongo_params += " --db " + args["db"]
    if args["collection"]:
            mongo_params += " --collection " + args["collection"]
    return mongo_params


def main():
    args = vars(parser.parse_args())

    # Configure action
    if args["action"] == "configure":
        cfg_parser = ConfigParser.SafeConfigParser()
        cfg_parser.add_section("aws")
        cfg_parser.set("aws", "access_key", raw_input("AWS Access Key: "))
        cfg_parser.set("aws", "secret_key", raw_input("AWS Secret Key: "))
        cfg_parser.set("aws", "s3_bucket", raw_input("S3 Bucket Name: "))
        cfg_parser.set("aws", "folder", raw_input("folder: "))

        cfg_parser.add_section("mongodb")
        cfg_parser.set("mongodb", "host", raw_input("MongoDB Host: "))
        cfg_parser.set("mongodb", "username", raw_input("MongoDB Username: "))
        cfg_parser.set("mongodb", "password", raw_input("MongoDB Password: "))
        cfg_parser.write(open(os.path.expanduser("~/.mo2s3.cfg"), "w"))

        print "Config written in %s" % os.path.expanduser("~/.mo2s3.cfg")
        sys.exit(0)

    if not args["access_key"] or not args["secret_key"]:
        print "S3 credentials not set, run 'mo2s3 configure' or specify --access-key/--secret-key, 'mo2s3 -h' to show the help"
        sys.exit(0)

    # Amazon S3 connection
    conn = boto.connect_s3(args["access_key"], args["secret_key"])
    bucket = conn.get_bucket(args["bucket"])
    folder = args["folder"]

    now = datetime.now().strftime("%Y%m%d%H%M%S")

    print "S3 Bucket: " + args["bucket"]
    print "S3 Folder: " + args["folder"]

    def key_for_filename(filename):
        if folder:
            return folder + '/' + filename
        else:
            return filename

    # Backup action
    if args["action"] == "backup":
        print "MongoDB: " + args["host"]

        # Run mongodump
        dump_directory = "mongodump_" + now
        mongo_cmd = "mongodump" + make_mongo_params(args) + " -o " + dump_directory
        if not args["db"]:
            mongo_cmd += " --oplog"
        print mongo_cmd

        mongodump = envoy.run(mongo_cmd)
        if mongodump.status_code != 0:
            print mongodump.std_err
        print mongodump.std_out

        # Create dump archive
        tar_filename = "mongodump"
        if args["db"]:
            tar_filename += "_" + args["db"]
        if args["collection"]:
            tar_filename += "_" + args["collection"]
        tar_filename += "_" + now + ".tgz"
        tar = envoy.run("tar czf " + tar_filename + " " + dump_directory)
        if tar.status_code != 0:
            print tar.std_err
        print tar.std_out

        # Upload to S3
        k = Key(bucket)
        k.key = key_for_filename(tar_filename)
        k.set_contents_from_filename(tar_filename)

        print tar_filename + " uploaded"
        os.remove(tar_filename)
        try:
            shutil.rmtree(dump_directory)
        except Exception:
            print "shutil.rmtree({dump_directory}) got an error (does {dump_directory} really exist?)"\
                .format(dump_directory=dump_directory)

    # List action
    elif args["action"] == "list":
        # List bucket files
        for key in bucket.list(prefix=folder):
                print key.name
        # print folder
        # if folder:
        #     for key in bucket.list(prefix=folder):
        #         print key.name
        # else:
        #     for key in bucket.get_all_keys():
        #         print key.name

    # Drop action
    elif args["action"] == "drop":
        # Delete every file in the bucket
        for key in bucket.list(folder):
            print "deleting " + key.name
            key.delete()

    # Delete action
    elif args["action"] == "delete":
        # Delete the backup on S3
        if not args["filename"]:
            print "No filename specified (--filename), 'mo2s3 -h' to show the help"
            sys.exit(0)
        k = Key(bucket)
        k.key = key_for_filename(args["filename"])
        print "deleting " + args["filename"]
        k.delete()

    # Restore action
    elif args["action"] == "restore":
        print "MongoDB: " + args["host"]

        if not args["filename"]:
            print "No filename specified (--filename), 'mo2s3 -h' to show the help"
            sys.exit(0)

        # Download backup file from S3
        k = Key(bucket)
        k.key = key_for_filename(args["filename"])
        print "restoring " + args["filename"]
        k.get_contents_to_filename(args["filename"])
        dump_date = args["filename"][-18:-4]
        tar = envoy.run("tar xvzf " + args["filename"])
        if tar.status_code != 0:
            print tar.std_err
        print tar.std_out

        # Run mongorestore
        restore_cmd = "mongorestore" + make_mongo_params(args) + " mongodump_" + dump_date
        if args["db"]:
            restore_cmd += "/" + args["db"]
            if args["collection"]:
                restore_cmd += "/" + args["collection"]
        else:
            restore_cmd += " --oplogReplay"
        print restore_cmd
        # mongorestore = envoy.run(restore_cmd)
        # if mongorestore.status_code != 0:
        #     print mongorestore.std_err
        # print mongorestore.std_out
        #
        # # Remove generated file
        # dump_directory = "mongodump_" + dump_date
        # try:
        #     shutil.rmtree(dump_directory)
        # except Exception:
        #     print "shutil.rmtree({dump_directory}) got an error (does {dump_directory} really exist?)"\
        #         .format(dump_directory=dump_directory)
        # os.remove(args["filename"])


if __name__ == '__main__':
    main()
