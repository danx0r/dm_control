#!/usr/bin/env python3
"""Demo script showing inverse kinematics with robotic arm visualization."""

import numpy as np
import mujoco.viewer
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_control.mujoco.testing import assets
from dm_control.utils import inverse_kinematics as ik
import time

mjlib = mjbindings.mjlib

# Constants
SITE_NAME = 'gripsite'
JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
TARGET_SPHERE_SIZE = 0.03
ANIMATION_STEPS = 100

class IKDemo:
    def __init__(self):
        # Target position (initialize before using in XML modification)
        self.target_pos = np.array([0.3, 0.2, 0.4])
        
        # Load the arm model
        arm_xml = assets.get_contents('arm.xml').decode('utf-8')
        
        # Add a target sphere to the XML
        modified_xml = self._add_target_sphere(arm_xml)
        self.physics = mujoco.Physics.from_xml_string(modified_xml)
        
        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.physics.model.ptr, self.physics.data.ptr)
        
    def _add_target_sphere(self, xml_string):
        """Add a target sphere and checkerboard floor to the XML."""
        # Insert the target sphere and checkerboard floor before the closing worldbody tag
        additions = f'''
    <body name='target' pos='{self.target_pos[0]:.3f} {self.target_pos[1]:.3f} {self.target_pos[2]:.3f}'>
      <geom name='target_sphere' type='sphere' size='{TARGET_SPHERE_SIZE}' rgba='1 0 0 0.7' contype='0' conaffinity='0'/>
    </body>
    <body name='floor' pos='0 0 0'>
      <geom name='floor' type='plane' size='2 2 0.1' rgba='0.8 0.9 0.8 0.2' material='grid'/>
    </body>
    <light name='key_light' pos='1 1 2' dir='-1 -1 -2' diffuse='0.8 0.8 0.8' specular='0.3 0.3 0.3'/>
    <light name='fill_light' pos='-1 1 1.5' dir='1 -1 -1.5' diffuse='0.4 0.4 0.4' specular='0.1 0.1 0.1'/>
    <light name='back_light' pos='0 -1.5 1' dir='0 1.5 -1' diffuse='0.3 0.3 0.3' specular='0.1 0.1 0.1'/>'''
        
        # Add checkerboard material to assets section if it exists, otherwise add asset section
        if '<asset>' in xml_string:
            # Insert material into existing asset section
            asset_insertion = '''
    <material name='grid' texture='grid' texrepeat='8 8' rgba='0.2 0.3 0.2 0.2'/>
    <texture name='grid' type='2d' builtin='checker' width='512' height='512' rgb1='0.1 0.2 0.3' rgb2='0.2 0.3 0.4'/>'''
            xml_string = xml_string.replace('</asset>', asset_insertion + '\n  </asset>')
        else:
            # Add entire asset section after the compiler section
            asset_section = '''
  <asset>
    <material name='grid' texture='grid' texrepeat='8 8' rgba='0.2 0.3 0.2 0.2'/>
    <texture name='grid' type='2d' builtin='checker' width='512' height='512' rgb1='0.1 0.2 0.3' rgb2='0.2 0.3 0.4'/>
  </asset>
'''
            # Insert after the compiler section
            if '<compiler' in xml_string:
                compiler_end = xml_string.find('/>', xml_string.find('<compiler')) + 2
                xml_string = xml_string[:compiler_end] + '\n\n' + asset_section + xml_string[compiler_end:]
            else:
                # Insert after the opening mujoco tag
                mujoco_end = xml_string.find('>', xml_string.find('<mujoco')) + 1
                xml_string = xml_string[:mujoco_end] + '\n' + asset_section + xml_string[mujoco_end:]
        
        # Find the closing worldbody tag and insert before it
        closing_tag = '</worldbody>'
        xml_with_additions = xml_string.replace(closing_tag, additions + '\n  ' + closing_tag)
        return xml_with_additions
    
    def set_random_arm_configuration(self):
        """Set the arm to a random configuration."""
        # Get joint limits
        joint_ranges = self.physics.named.model.jnt_range[JOINTS]
        limited = self.physics.named.model.jnt_limited[JOINTS].astype(bool)
        
        # Generate random joint positions within limits
        qpos = np.zeros(len(JOINTS))
        for i, joint in enumerate(JOINTS):
            if limited[i]:
                low, high = joint_ranges[i]
                qpos[i] = np.random.uniform(low, high)
            else:
                qpos[i] = np.random.uniform(0, 2 * np.pi)
        
        # Set joint positions
        self.physics.named.data.qpos[JOINTS] = qpos
        self.physics.forward()
    
    def solve_ik(self, target_pos):
        """Solve inverse kinematics for the given target position."""
        result = ik.qpos_from_site_pose(
            physics=self.physics,
            site_name=SITE_NAME,
            target_pos=target_pos,
            joint_names=JOINTS,
            tol=1e-12,
            max_steps=100,
            inplace=False
        )
        return result
    
    def animate_to_target(self, target_qpos):
        """Animate the arm from current position to target position."""
        start_qpos = self.physics.named.data.qpos[JOINTS].copy()
        
        for step in range(ANIMATION_STEPS + 1):
            # Interpolate between start and target positions
            alpha = step / ANIMATION_STEPS
            current_qpos = (1 - alpha) * start_qpos + alpha * target_qpos
            
            # Set the joint positions
            self.physics.named.data.qpos[JOINTS] = current_qpos
            self.physics.forward()
            
            # Update viewer
            if self.viewer.is_running():
                self.viewer.sync()
                time.sleep(0.02)  # 50 FPS
            else:
                break
    
    def update_target_position(self, new_pos):
        """Update the target sphere position."""
        self.target_pos = new_pos.copy()
        self.physics.named.model.body_pos['target'] = self.target_pos
        self.physics.forward()
    
    def run_demo(self):
        """Run the interactive demo."""
        print("IK Demo Controls:")
        print("- Press 'r' to randomize arm position")
        print("- Press 's' to solve IK and animate to target")
        print("- Press 't' to set new random target")
        print("- Press 'q' to quit")
        print()
        
        # Set initial random configuration
        self.set_random_arm_configuration()
        
        while self.viewer.is_running():
            # Get current end-effector position
            current_pos = self.physics.named.data.site_xpos[SITE_NAME].copy()
            distance = np.linalg.norm(current_pos - self.target_pos)
            
            print(f"\rEnd-effector distance to target: {distance:.4f}m", end="", flush=True)
            
            # Check for keyboard input (simplified - in real application you'd use proper input handling)
            key = input("\nEnter command (r/s/t/q): ").lower().strip()
            
            if key == 'r':
                print("Randomizing arm position...")
                self.set_random_arm_configuration()
                
            elif key == 's':
                print("Solving IK and animating to target...")
                result = self.solve_ik(self.target_pos)
                
                if result.success:
                    print(f"IK solved in {result.steps} steps with error {result.err_norm:.6f}")
                    self.animate_to_target(result.qpos[self.physics.named.model.jnt_qposadr[JOINTS]])
                else:
                    print("IK failed to converge!")
                    
            elif key == 't':
                print("Setting new random target...")
                # Generate random target within reachable workspace
                new_target = np.array([
                    np.random.uniform(-0.6, 0.6),
                    np.random.uniform(-0.6, 0.6),
                    np.random.uniform(0.1, 0.8)
                ])
                self.update_target_position(new_target)
                print(f"New target: [{new_target[0]:.3f}, {new_target[1]:.3f}, {new_target[2]:.3f}]")
                
            elif key == 'q':
                break
                
        self.viewer.close()

def main():
    """Main function to run the demo."""
    print("Starting IK Demo...")
    demo = IKDemo()
    demo.run_demo()
    print("Demo completed.")

if __name__ == '__main__':
    main()