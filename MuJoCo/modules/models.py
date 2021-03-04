# [Built-in modules]
import os
import re
import sys
import shutil
import time, datetime
from   random import randint, seed

# [3rd party modules]
import numpy                 as np
import xml.etree.ElementTree as ET

# [Local modules]
from modules.constants import Constants



class TransmissionLine:
    """
        Description:
        ----------

        Arguments:
        ----------

        Returns:
        ----------
    """
    def __init__( self, m = 1, k = 1, b = 0, N = 50 ):
        # N masses, N-1 springs
        # Assuming a uniform mass/spring/damper value along the transmission line.
        self.m = m
        self.k = k
        self.b = b                                                              # [TODO] We are not considering lossy transmission line temporarily.
        self.N = N                                                              # Number of sub-models for the transmission line


        seed( datetime.datetime.now( ) )                                        # Seed randomization.

        # The propagation constant and characteristic impedance of the model
        # [REF] https://arxiv.org/pdf/1309.6898.pdf

        self.gamma = np.sqrt( self.m / self.k )                                 # The propagation constant. The wave speed is inverse proportional to this, meaning, if k increases or m decreases, then wave speed increase.
        self.Z0    = np.sqrt( self.m * self.k )                                 # The characteristic impedance of the transmision line


    def gen_mass( self, N, pos = 0, name = None, m = None ):
        return ET.Element( "geom", attrib = {
                                    "type" : "box"                                       ,
                                    "name" : "mass" + str( N ) if not name else name     ,
                               	    "mass" : str( self.m )     if not    m else self.m   ,
                                    "pos"  : "{0:7.4f} 0 0".format( pos )                ,
								"material" : "JointColor"                                ,
                                    "size" : "{0:7.4f}".format( 0.1/self.N ) * 3 }       )

    def gen_spring( self, N, pos = 0, name = None, k = None ):
        return ET.Element( "joint", attrib = {
                                    "ref"  : "0"             	                         ,
                                    "type" : "slide"      	                             ,
                                    "name" : "spring" + str( N ) if not name else name   ,
                               "stiffness" : str( self.k )       if not    k else    k   ,
                                    "pos"  : "{0:7.4f} 0 0".format( pos )                ,
                                    "axis" : "1 0 0" } )

    def gen_body( self, N, d = 0.1 ):
        return ET.Element( "body",  attrib = {
                            "name" : "node" + str( N + 1 )         ,
                           "euler" : "0 0 0"                       ,
                            "pos"  : "{0:7.4f} 0 0".format( 0.5/self.N ) } )

    def gen_submodel( self, N, p_elem ):

        p_elem.append( self.gen_spring( N, name = "sender"  if N == 1       else None )  )   # Attaching the spring, which is a joint. The joint is connected between the body where it is defined and its parent body.
        p_elem.append( self.gen_mass(   N, name = "tip"     if N == self.N  else None )  )

        if N == self.N:
            return

        n_elem = self.gen_body(  N )
        p_elem.append( n_elem )

        self.gen_submodel( N + 1, n_elem )                                      # Recursively generating the model

    def gen_xml( self ):
        """
            Description
            -----------

                Generating the .xml model file for MuJoCo

        """

        model_dir       = Constants.MODEL_DIR
        my_model_tree   = ET.parse( model_dir + "/root.xml" )                   # Importing/parsing the mujoco mjcf xml template file to customize and generate the N-Node whip model.
        root            = my_model_tree.getroot( )
                                                                                # [REF] https://stackoverflow.com/questions/2170610/access-elementtree-node-parent-nod
		# Append the transmission line model to the "sender" actuator
        for elem in root.iter( 'body' ):
            if 'root' in elem.attrib.values( ):
                self.gen_submodel( 1, elem )

        self.clean_up( root )                                                   # Cleaning up the XML file

        self.name = "TL_Line_{0:d}.xml".format( self.N )

        my_model_tree.write( model_dir + "/" + self.name,
                         encoding = "utf-8" ,
                  xml_declaration = False   )


    def clean_up( self, elem, level = 0 ):                                      # Basic Clean up function for the xml model file, adding newline and indentation.
                                                                                # [REF] https://norwied.wordpress.com/2013/08/27/307/
        i = "\n" + level * "   "

        if len( elem ):
            if not elem.text or not elem.text.strip( ):
                elem.text = i + "  "

            if not elem.tail or not elem.tail.strip( ):
                elem.tail = i

            for elem in elem:
                self.clean_up( elem, level + 1 )

            if not elem.tail or not elem.tail.strip( ):
                elem.tail = i
        else:
            if level and ( not elem.tail or not elem.tail.strip( ) ):
                elem.tail = i
