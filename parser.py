
def parse_file(filepath = 'outputs/13-NX ABSOLUTE/F27 - Do it!/F27 - Do it!.ssc'):
    data = {}

    with open(filepath, 'r') as f:

        while True:
        
            line = f.readline().split('//')[0].strip()
            
            if 'NOTES' in line:
                notes = []
                compas = []
                line = f.readline().split('//')[0].strip()
                while line != ';':
                    if ',' not in line:
                        compas.append(line)
                    else:
                        notes.append(compas)
                        compas = []
                    line = f.readline().split('//')[0].strip()
                data['NOTES'] = notes
                break # toma la primera

            else:
                while ';' not in line:
                    line = line.split('//')[0].strip()
                    line += f.readline().strip()
                
                line = line.split(';')[0]
                line = line.replace('#', '')

                name, val = line.split(':')

                data[name] = val.strip()

    return data

parse_file()