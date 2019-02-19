from inputs import get_gamepad
from inputs import get_key
import time
import datetime
import os
import csv
import pyscreenshot as screenshot

button_dictionary = {'BTN_SOUTH1': 'A',
                     'BTN_WEST1': 'B',
                     'ABS_HAT0Y-1': 'UP',
                     'ABS_HAT0Y1': 'DOWN',
                     'ABS_HAT0X1': 'RIGHT',
                     'ABS_HAT0X-1': 'LEFT'}
# [image,    up,down,left,right,A,B]
# ['filename' 0,  0,    0,   0,  0,0]


def gamepad_buttons(csv_writer, directory):
    done = True
    counter = 0
    row = [['', 0, 0, 0, 0, 0, 0]]
    while done:

        events = 12

        # image_directory = directory + "\\" + str(counter) + ".png"
        # image_directory = str(counter) + ".png"
        image = screenshot.grab(bbox=(0, 56, 1172, 898))
        while events is not None and done:
            try:
                events = get_gamepad(False)
            except RuntimeError:
                events = None
            if events is not None:
                event = events[0]
                if event.code == 'BTN_TR':
                    done = False
                if event.code == 'ABS_HAT0Y' or event.code == 'ABS_HAT0X' or event.code == 'BTN_SOUTH' or \
                        event.code == 'BTN_WEST' or 'BTN_TR':
                    # UP
                    if event.code == 'ABS_HAT0Y' and event.state == -1:
                        row[0][1] = 1
                    elif event.code == 'ABS_HAT0Y' and event.state == 0:
                        row[0][1] = 0
                    # DOWN
                    if event.code == 'ABS_HAT0Y' and event.state == 1:
                        row[0][2] = 1
                    elif event.code == 'ABS_HAT0Y' and event.state == 0:
                        row[0][2] = 0
                    # LEFT
                    if event.code == 'ABS_HAT0X' and event.state == -1:
                        row[0][3] = 1
                    elif event.code == 'ABS_HAT0X' and event.state == 0:
                        row[0][3] = 0
                    # RIGHT
                    if event.code == 'ABS_HAT0X' and event.state == 1:
                        row[0][4] = 1
                    elif event.code == 'ABS_HAT0X' and event.state == 0:
                        row[0][4] = 0
                    # A
                    if event.code == 'BTN_SOUTH' and event.state == 1:
                        row[0][5] = 1
                    elif event.code == 'BTN_SOUTH' and event.state == 0:
                        row[0][5] = 0
                    # B
                    if event.code == 'BTN_WEST' and event.state == 1:
                        row[0][6] = 1
                    elif event.code == 'BTN_WEST' and event.state == 0:
                        row[0][6] = 0
        image_directory = str(counter) + ".png"
        row[0][0] = image_directory
        image_directory = directory + "\\" + str(counter) + ".png"
        image.resize((586, 449))
        image.save(image_directory)
        print(row)
        csv_writer.writerow(row)
        # print(row)
        counter += 1


if __name__ == '__main__':

    # while 1:
    #     key_events = get_key()
    #     if key_events:
    #         for key in key_events:
    #             print(key.ev_type, key.code, key.state)
    directory = os.getcwd() + "\\capture_" + str(datetime.date.today()) + str(time.time())
    if not os.path.isdir(directory):
        os.makedirs(directory)

    csv.register_dialect('myDialect', delimiter='|', quoting=csv.QUOTE_NONE, skipinitialspace=True)
    csv_file_name = directory + "\\controller_capture.csv"
    csv_file_name = bytes(csv_file_name, 'utf8')
    with open(csv_file_name, 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        writer.writerow([['image', 'up', 'down', 'left', 'right', 'A', 'B']])
        gamepad_buttons(writer, directory)
    f.close()
