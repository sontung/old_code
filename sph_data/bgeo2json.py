#!/usr/bin/env python

"""
Converts partio files to and from json.

Usage:  partjson [FLAGS] <foo.(json|bgeo)> <bar.(bgeo|json)>

Supported FLAGS:
    -c/--compress: When converting to partio, compress the output
    -v/--verbose : Turn on verbosity for partio
    -h/--help    : Print this help message
"""

# TODO: Unicode compliance

__copyright__ = """
CONFIDENTIAL INFORMATION: This software is the confidential and
proprietary information of Walt Disney Animation Studios ("WDAS").
This software may not be used, disclosed, reproduced or distributed
for any purpose without prior written authorization and license
from WDAS.  Reproduction of any section of this software must
include this legend and all copyright notices.
Copyright Disney Enterprises, Inc.  All rights reserved.
"""


import sys
import glob
dir1 = glob.glob("../libraries/partio/build/*")
dir2 = [du for du in glob.glob("../libraries/partio/build/*/compiled/lib/*") if "python3." in du]
assert len(dir1) == 1 and len(dir2) == 1
sys.path.append(f"{dir2[0]}/site-packages")
print(f"detecting partio at {dir2[0]}/site-packages")
import sys, json
import partio
import glob


def toJson(particleSet):
    """ Converts a particle set to json """

    data = {}

    # Put types in json just for readability
    data['__types__'] = { i : partio.TypeName(i) for i in range(5) }

    # Convert fixed attributes
    fixedAttributes = {}
    fixedIndexedStrings = {}
    for i in range(particleSet.numFixedAttributes()):
        attr = particleSet.fixedAttributeInfo(i)
        fixedAttributes[attr.name] = {'type': attr.type,
                                      'count': attr.count,
                                      'value': particleSet.getFixed(attr),
                                     }

        # Convert indexed string attributse
        if attr.type == partio.INDEXEDSTR:
            fixedIndexedStrings[attr.name] = particleSet.fixedIndexedStrs(attr)

    if fixedAttributes:
        data['fixedAttributes'] = fixedAttributes
    if fixedIndexedStrings:
        data['fixedIndexedStrings'] = fixedIndexedStrings

    # Convert particle attributes
    attributes = {}
    attrs = []
    indexedStrings = {}
    for i in range(particleSet.numAttributes()):
        attr = particleSet.attributeInfo(i)
        attrs.append(attr)
        attributes[attr.name] = {'type': attr.type, 'count': attr.count }

        # Convert indexed string attributse
        if attr.type == partio.INDEXEDSTR:
            indexedStrings[attr.name] = particleSet.indexedStrs(attr)

    if attributes:
        data['attributes'] = attributes
    if indexedStrings:
        data['indexedStrings'] = indexedStrings

    # Convert particles to an indexed dictionary
    particles = {}
    for i in range(particleSet.numParticles()):
        particle = {}
        for attr in attrs:
            particle[attr.name] = particleSet.get(attr, i)
        # Add an index purely for readability & debugging (not consumed converting back)
        particles[i] = particle

    if particles:
        data['particles'] = particles

    return data


def fromJson(data):
    """ Converts a json dictionary to a particle set """

    particleSet = partio.create()

    # Convert fixed attributes
    fixedAttributes = {}
    if 'fixedAttributes' in data:
        for attrName, attrInfo in data['fixedAttributes'].items():
            attrName = str(attrName)
            attr = particleSet.addFixedAttribute(attrName, attrInfo['type'], attrInfo['count'])
            fixedAttributes[attrName] = attr
            if len(attrInfo['value']) == attrInfo['count']:
                particleSet.setFixed(attr, attrInfo['value'])
            else:
                sys.stderr.write('Mismatched count for fixed attribute {}. Skipping.\n'.format(attrName))

    # Convert attributes
    attributes = {}
    if 'attributes' in data:
        for attrName, attrInfo in data['attributes'].items():
            attrName = str(attrName)
            attr = particleSet.addAttribute(attrName, attrInfo['type'], attrInfo['count'])
            attributes[attrName] = attr

    # Convert fixed indexed strings
    if 'fixedIndexedStrings' in data:
        for attrName, strings in data['fixedIndexedStrings'].items():
            if attrName not in fixedAttributes:
                sys.stderr.write('Could not match fixed indexed string {} with any defined fixed attribute. Skipping.\n'.format(attrName))
                continue
            for string in strings:
                particleSet.registerFixedIndexedStr(fixedAttributes[attrName], string)

    # Convert indexed strings
    if 'indexedStrings' in data:
        for attrName, strings in data['indexedStrings'].items():
            if attrName not in attributes:
                sys.stderr.write('Could not match indexed string {} with any defined attribute. Skipping.\n'.format(attrName))
                continue
            for string in strings:
                particleSet.registerIndexedStr(attributes[attrName], str(string))

    # Convert particles
    if 'particles' in data:
        particleSet.addParticles(len(data['particles']))
        for pnum, particle in data['particles'].items():
            pnum = int(pnum)
            for attrName, value in particle.items():
                try:
                    attr = attributes[attrName]
                except IndexError:
                    sys.stderr.write('Could not match attribute "{}" for particle {} with any defined attributes. Skipping.\n'.format(attrName, pnum))
                    continue
                if len(value) != attr.count:
                    sys.stderr.write('Mismatched count for attribute "{}" ({}) and particle {} ({}). Skipping.\n'.format(attrName, attr.count, pnum, len(value)))
                    continue

                particleSet.set(attr, pnum, value)

    return particleSet

def process_bgeo2json():
    """ Main """

    # Process command-line arguments
    filenames = []
    verbose = False
    compress = False
    

    sph_states = "../data_heavy/sph_solutions/state/*.bgeo"
    all_files = glob.glob(sph_states)
    partio_extensions = ('.bgeo', '.geo', '.bhclassic', '.ptc', '.pdb')

    for file1 in all_files:
        particleSet = partio.read(file1, verbose)
        data = toJson(particleSet)
        with open("%s.json" % file1, 'w') as fp:
            json.dump(data, fp, indent=2, sort_keys=True)



if __name__ == '__main__':
    process_bgeo2json()
