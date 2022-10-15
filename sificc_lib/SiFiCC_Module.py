# Author: Awal Awal
# Date: May 2020
# Email: awal.nova@gmail.com

class SiFiCC_Module:
    '''Represents a single module (scatterer or absorber) within the SiFi-CC
    '''
    
    def __init__(self, thickness_x, thickness_y, thickness_z, position, orientation=0):
        self.thickness_x = thickness_x
        self.thickness_y = thickness_y
        self.thickness_z = thickness_z
        self.position = position
        self.orientation = orientation
        
        self.start_x = self.position.x - self.thickness_x/2
        self.end_x = self.position.x + self.thickness_x/2
        self.start_y = self.position.y - self.thickness_y/2
        self.end_y = self.position.y + self.thickness_y/2
        self.start_z = self.position.z - self.thickness_z/2
        self.end_z = self.position.z + self.thickness_z/2

    def is_cluster_inside(self, cluster):
        """
        check if given cluster is inside the module
        """
        if self.start_x < cluster.x < self.end_x:
            if self.start_y < cluster.y < self.end_y:
                if self.start_z < cluster.z < self.end_z:
                    return True
        return False

    def is_any_cluster_inside(self, list_cluster):
        """
        checks if any point is inside the scatterer/absorber volume
        """
        # iterate all clusters in the list
        for cluster in list_cluster:
            if self.is_cluster_inside(cluster):
                return True
        return False
