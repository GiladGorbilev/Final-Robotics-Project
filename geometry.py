class GeometricObject:
    def __init__(self, shape_type, x, y, z, width, height, length, name="Obj"):
        self.shape_type = shape_type
        # Coordinates and Dimensions
        self.x, self.y, self.z = x, y, z
        self.width, self.height, self.length = width, height, length
        self.name = name

        # Helper for plotting boundaries (used in main.py)
        # Note: We maintain x_min/max style for simple bounding box plotting if needed
        self.x_min, self.x_max = sorted([x, x + length])
        self.y_min, self.y_max = sorted([y, y + width])
        self.z_min, self.z_max = sorted([z, z + height])

    def contains_point(self, px, py, pz):
        # 1. BLOCK LOGIC
        if self.shape_type == 'block':
            return (self.x_min <= px <= self.x_max) and \
                   (self.y_min <= py <= self.y_max) and \
                   (self.z_min <= pz <= self.z_max)

        # 2. TUBE LOGIC (Center-based XY, Start-based Z)
        elif self.shape_type == 'tube':
            # Length Check (along X-axis)
            # Depending on definition, length might be along Z or X. 
            # Snippet 2 implies length is X-based for the tube check:
            if not (self.x <= px <= (self.x + self.length)):
                return False

            # Circle Check (in YZ plane)
            dy = py - self.y
            dz = pz - self.z
            dist_sq = dy*dy + dz*dz

            # Width = Outer Diameter, Height = Inner Diameter
            r_outer_sq = (self.width / 2) ** 2
            r_inner_sq = (self.height / 2) ** 2

            return r_inner_sq <= dist_sq <= r_outer_sq

        return False