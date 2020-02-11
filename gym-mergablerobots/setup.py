from setuptools import setup

setup(name='gym_mergablerobots',
      version='0.0.1',
      install_requires=['gym'],  # And any other dependencies foo needs
      package_data={'gym': [
          'envs/assets/*.xml',
          'envs/assets/meshes/*.stl',
          'envs/assets/textures/*.png']}
      )
