import os
import glob

def main():
    ann_dir = '/data/VTD/LB/Annotations'
    apt_files = glob.glob(os.path.join(ann_dir, 'apartment', '*'))
    hotel_files = glob.glob(os.path.join(ann_dir, 'hotel', '*'))
    office_files = glob.glob(os.path.join(ann_dir, 'office', '*'))

    apt_spks = []
    for ff in apt_files:
        f = open(ff, 'r')
        ll = f.readlines()
        apt_spks = apt_spks + [l.strip().split(' ')[2] for l in ll]
        f.close()
    hotel_spks = []
    for ff in hotel_files:
        f = open(ff, 'r')
        ll = f.readlines()
        hotel_spks = hotel_spks + [l.strip().split(' ')[2] for l in ll]
        f.close()
    office_spks = []
    for ff in office_files:
        f = open(ff, 'r')
        ll = f.readlines()
        office_spks = office_spks + [l.strip().split(' ')[2] for l in ll]
        f.close()

    apt_set = set(apt_spks)
    hotel_set = set(hotel_spks)
    office_set = set(office_spks)

    print('All speakers')
    all_spks = list(apt_set.union(hotel_set.union(office_set)))
    all_spks.sort()
    print(all_spks)

    print('\nSpeakers in all rooms')
    shared_spks = list(apt_set.intersection(hotel_set.intersection(office_set)))
    shared_spks.sort()
    print(shared_spks)

    print('\nSpeakers that appear in the apartment and the hotel but not the office')
    apt_hotel_spks = list(apt_set.intersection(hotel_set)-office_set)
    apt_hotel_spks.sort()
    print(apt_hotel_spks)

    print('\nSpeakers that appear in the apartment and the office but not the hotel')
    apt_office_spks = list(apt_set.intersection(office_set)-hotel_set)
    apt_office_spks.sort()
    print(apt_office_spks)

    print('\nSpeakers that appear in the apartment and the office but not the hotel')
    hotel_office_spks = list(hotel_set.intersection(office_set)-office_set)
    hotel_office_spks.sort()
    print(hotel_office_spks)

    print('\nSpeakers that appear only in the apartment')
    apt_spks = list(apt_set-hotel_set-office_set)
    apt_spks.sort()
    print(apt_spks)

    print('\nSpeakers that appear only in the hotel')
    hotel_spks = list(hotel_set-apt_set-office_set)
    hotel_spks.sort()
    print(hotel_spks)

    print('\nSpeakers that appear only in the office')
    office_spks = list(office_set-hotel_set-apt_set)
    office_spks.sort()
    print(office_spks)

if __name__=='__main__':
    main()