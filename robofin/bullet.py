from pathlib import Path
import time
import copy
import numpy as np
import pybullet as p
from geometrout import Cuboid, Sphere, Cylinder, SE3
from scipy.spatial.transform import Rotation as R
from robofin.robots import FrankaRobot, FrankaGripper
from robofin.pointcloud.numpy import transform_pointcloud
import math
import trimesh
from ikfast_franka_panda import get_fk, get_ik

class BulletRobot:
    def __init__(self, clid, hd=False, **kwargs):
        self.clid = clid
        self.hd = hd
        # TODO implement collision free robot
        self.id = self.load(clid)
        self._setup_robot()
        self._set_robot_specifics(**kwargs)
        self._set_gripper_constraint()
        self._set_robot_initial_pose(np.r_[-6.8229338e-06, -1.9273859e-01, 1.1898350e-03, -1.9625933e+00,
                                           1.4666162e-03, 1.7754314e+00, 7.8662562e-01, 0.04, 0.04])

    def load(self, clid, urdf_path=None):
        if self.hd:
            urdf = self.robot_type.hd_urdf
        else:
            urdf = self.robot_type.urdf
        return p.loadURDF(urdf,
                          [0, 0, 0.0],
                          useFixedBase=True,
                          physicsClientId=clid,
                          flags=p.URDF_USE_SELF_COLLISION)

    @property
    def links(self):
        return [(k, v) for k, v in self._link_name_to_index.items()]

    def link_id(self, name):
        return self._link_name_to_index[name]

    def link_name(self, id):
        return self._index_to_link_name[id]

    @property
    def link_frames(self):
        ret = p.getLinkStates(self.id,
                              list(range(len(self.links) - 1)),
                              computeForwardKinematics=True,
                              physicsClientId=self.clid)
        frames = {}
        for ii, r in enumerate(ret):
            frames[self.link_name(ii)] = SE3(np.array(r[4]),
                                             np.array([r[5][3], r[5][0], r[5][1], r[5][2]]))
        return frames

    def closest_distance_to_self(self, max_radius):
        contacts = p.getClosestPoints(self.id, self.id, max_radius, physicsClientId=self.clid)
        # Manually filter out fixed connections that shouldn't be considered
        # TODO fix this somehow
        filtered = []
        for c in contacts:
            # A link is always in collision with itself and its neighbors
            if abs(c[3] - c[4]) <= 1:
                continue
            # panda_link8 just transforms the origin
            if c[3] == 6 and c[4] == 8:
                continue
            if c[3] == 8 and c[4] == 6:
                continue
            if c[3] > 8 or c[4] > 8:
                continue
            filtered.append(c)
        if len(filtered):
            return min([x[8] for x in filtered])
        return None

    def closest_distance(self, obstacles, max_radius, ignore_base=True):
        distances = []
        for id in obstacles:
            closest_points = p.getClosestPoints(
                self.id, id, max_radius, physicsClientId=self.clid
            )
            if ignore_base:
                closest_points = [c for c in closest_points if c[3] > -1]
            distances.extend([x[8] for x in closest_points])
        if len(distances):
            return min(distances)
        return None

    def in_collision(self, obstacles, radius=0.0, check_self=False, ignore_base=True):
        raise NotImplementedError

    def get_collision_points(self, obstacles, check_self=False):
        points = []
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        if check_self:
            contacts = p.getContactPoints(self.id, self.id, physicsClientId=self.clid)
            # Manually filter out fixed connections that shouldn't be considered
            # TODO fix this somehow
            filtered = []
            for c in contacts:
                # panda_link8 just transforms the origin
                if c[3] == 6 and c[4] == 8:
                    continue
                if c[3] == 8 and c[4] == 6:
                    continue
                if c[3] > 8 or c[4] > 8:
                    continue
                filtered.append(c)
            points.extend([p[5] for p in filtered])

        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getContactPoints(self.id, id, physicsClientId=self.clid)
            points.extend([p[5] for p in contacts])
        return points

    def get_deepest_collision(self, obstacles):
        distances = []
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getContactPoints(self.id, id, physicsClientId=self.clid)
            distances.extend([p[8] for p in contacts])

        if len(distances) > 0:
            # Distance will be negative if it's a true penetration
            deepest_collision = min(distances)
            if deepest_collision < 0:
                return abs(deepest_collision)
        return 0

    def get_collision_depths(self, obstacles):
        distances = []
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getContactPoints(self.id, id, physicsClientId=self.clid)
            distances.extend([p[8] for p in contacts])

        return [abs(d) for d in distances if d < 0]

    def _setup_robot(self):
        self._link_name_to_index = {p.getBodyInfo(self.id, physicsClientId=self.clid)[0].decode('UTF-8'): -1}
        for _id in range(p.getNumJoints(self.id, physicsClientId=self.clid)):
            _name = p.getJointInfo(self.id, _id, physicsClientId=self.clid)[12].decode('UTF-8')
            self._link_name_to_index[_name] = _id
        self._index_to_link_name = {}

        for k, v in self._link_name_to_index.items():
            self._index_to_link_name[v] = k

    def _set_robot_specifics(self, **kwargs):
        raise NotImplemented('Must be set in the robot specific class')

    def _set_gripper_constraint(self):
        raise NotImplemented('Must be set in the robot specific class')

    def _set_robot_initial_pose(self):
        raise NotImplemented('Must be set in the robot specific class')
    
    def update_debug_line(self, pos, quat, len, color=None, replaceItemUniqueId=None):
        mtx = np.eye(4)
        mtx[:3, :3] = R.from_quat(quat).as_matrix()
        pos_end = (np.r_[pos] + (mtx @ np.r_[0.0, 0.0, len, 1])[:3]).tolist()
        
        if replaceItemUniqueId is None:
            uid = p.addUserDebugLine(pos, pos_end, lineColorRGB=[1, 0, 0] if color is None else color,
                                     lineWidth=2.0, lifeTime=0, physicsClientId=self.clid)
        else:
            uid = p.addUserDebugLine(pos, pos_end, lineColorRGB=[1, 0, 0] if color is None else color,
                                     lineWidth=2.0, lifeTime=0, replaceItemUniqueId=replaceItemUniqueId,
                                     physicsClientId=self.clid)
        return uid

    def update_collision_debug_points(self, collisions, color=None, replaceItemUniqueId=None):
        if color is None:
            colors = [[1, 0, 0] for c in collisions]
        else:
            colors = [color for c in collisions]

        if replaceItemUniqueId is None:
            uid = p.addUserDebugPoints([list(c[1]) for c in collisions],
                                       pointColorsRGB=colors,
                                       pointSize=15.0, physicsClientId=self.clid)
        else:
            uid = p.addUserDebugPoints([list(c[1]) for c in collisions],
                                       pointColorsRGB=colors,
                                       pointSize=15.0, replaceItemUniqueId=replaceItemUniqueId,
                                       physicsClientId=self.clid)
        return uid
    
    def remove_collision_debug_points(self, itemUniqueId):
        p.removeUserDebugItem(itemUniqueId=itemUniqueId, physicsClientId=self.clid)
        

class BulletFranka(BulletRobot):
    robot_type = FrankaRobot

    def _set_robot_specifics(self, default_prismatic_value=0.025):
        self.default_prismatic_value = default_prismatic_value

    def _set_gripper_constraint(self):
        self.gcid = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=self.link_id('panda_leftfinger'),
            childBodyUniqueId=self.id,
            childLinkIndex=self.link_id('panda_rightfinger'),
            jointType=p.JOINT_GEAR,
            jointAxis=[1.0, 0.0, 0.0],
            parentFramePosition=np.r_[0., 0., 0.],
            parentFrameOrientation=np.r_[0., 0., 0., 1.],
            childFramePosition=np.r_[0., 0., 0.],
            childFrameOrientation=np.r_[0., 0., 0., 1.],
            physicsClientId=self.clid
        )
        p.changeConstraint(self.gcid, gearRatio=-1, erp = 0.1, maxForce=50)
    
    def _set_robot_initial_pose(self, state):
        velocities = [0.0 for _ in state]
        for i in range(0, 7):
            p.resetJointState(self.id,
                              i,
                              state[i],
                              targetVelocity=velocities[i],
                              physicsClientId=self.clid)
        p.resetJointState(self.id,
                          9,
                          state[7],
                          targetVelocity=velocities[7],
                          physicsClientId=self.clid)
        p.resetJointState(self.id,
                          10,
                          state[8],
                          targetVelocity=velocities[8],
                          physicsClientId=self.clid)


    def marionette(self, state, velocities=None, kinematics=False):
        if velocities is None:
            velocities = [0.0 for _ in state]
        assert len(state) == len(velocities)
        for i in range(0, 7):
            if kinematics:
                p.resetJointState(self.id, i, state[i],
                                  targetVelocity=velocities[i], physicsClientId=self.clid)
            else:
                p.setJointMotorControl2(self.id, i, p.POSITION_CONTROL, targetPosition=state[i],
                                        physicsClientId=self.clid)

        if len(state) == 9:
            if kinematics:
                p.resetJointState(self.id, 9, state[7],
                                  targetVelocity=velocities[7], physicsClientId=self.clid)
                p.resetJointState(self.id, 10, state[8],
                                  targetVelocity=velocities[8], physicsClientId=self.clid)
            else:
                p.setJointMotorControl2(self.id, 9, p.POSITION_CONTROL, targetPosition=state[7],
                                        force=20, physicsClientId=self.clid)
                p.setJointMotorControl2(self.id, 10, p.POSITION_CONTROL, targetPosition=state[8],
                                        force=20, physicsClientId=self.clid)

        elif len(state) == 7:
            p.resetJointState(self.id, 9, self.default_prismatic_value,
                              targetVelocity=0.0, physicsClientId=self.clid)
            p.resetJointState(self.id, 10, self.default_prismatic_value,
                              targetVelocity=0.0, physicsClientId=self.clid)
        else:
            raise Exception('Length of input state should be either 7 or 9')

    def get_joint_states(self, fk=False):
        states = p.getJointStates(self.id, [0, 1, 2, 3, 4, 5, 6, 9, 10], physicsClientId=self.clid)
        if not fk:
            return [s[0] for s in states], [s[1] for s in states]
        else:
            pos_fk, mtx_fk = get_fk([s[0] for s in states][:7])
            return [s[0] for s in states], [s[1] for s in states], pos_fk, mtx_fk
            
    def get_cartesian_state(self, name='panda_grasptarget'):
        uid = self.link_id(name)
        pos, quat = p.getLinkState(self.id, uid, physicsClientId=self.clid)[:2]
        return pos, quat

    def get_jacobian(self, name='panda_grasptarget'):
        uid = self.link_id(name)
        result = p.getLinkState(self.id,
                                uid,
                                computeLinkVelocity=1,
                                computeForwardKinematics=1,
                                physicsClientId=self.clid)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result

        joint_states = p.getJointStates(self.id,
                                        range(p.getNumJoints(self.id, physicsClientId=self.clid)),
                                        physicsClientId=self.clid)
        joint_infos = [p.getJointInfo(self.id, i, physicsClientId=self.clid)
                       for i in range(p.getNumJoints(self.id, physicsClientId=self.clid))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]

        mpos = [state[0] for state in joint_states]
        zero_vec = [0.0] * len(mpos)

        jac_t, jac_r = p.calculateJacobian(self.id, uid, com_trn, mpos, zero_vec, zero_vec, physicsClientId=self.clid)

        jacobian = np.concatenate([np.array(jac_t), np.array(jac_r)])
        return jacobian[:, :7]

    def control_position(self, state):
        assert len(state) in [7, 9]
        p.setJointMotorControlArray(
            self.id,
            jointIndices=list(range(len(state))),
            controlMode=p.POSITION_CONTROL,
            targetPositions=state,
            targetVelocities=[0] * len(state),
            forces=[250] * len(state),
            positionGains=[0.01] * len(state),
            velocityGains=[1.0] * len(state),
            physicsClientId=self.clid,
        )

    def in_collision(self, obstacles, radius=0.0, check_self=False, ignore_base=True):
        # Step the simulator (only enough for collision detection)
        if check_self:
            contacts = p.getContactPoints(
                self.id, self.id,
                # radius,
                physicsClientId=self.clid
            )
            # Manually filter out fixed connections that shouldn't be considered
            # TODO fix this somehow
            filtered = []
            for c in contacts:
                # A link is always in collision with itself and its neighbors
                if abs(c[3] - c[4]) <= 1:
                    continue
                # panda_link8 just transforms the origin
                if c[3] == 6 and c[4] == 8:
                    continue
                if c[3] == 8 and c[4] == 6:
                    continue
                if c[3] > 8 or c[4] > 8:
                    continue
                filtered.append(c)
            if len(filtered) > 0:
                return [[c[1], c[5], c[9]] for c in contacts]

        # Iterate through all obstacles to check for collisions
        for i, id in enumerate(obstacles[1:]):
            contacts = p.getContactPoints(
                self.id, id,
                # radius,
                physicsClientId=self.clid
            )
            if ignore_base:
                contacts = [c for c in contacts if c[3] > -1]
            if i == 0:
                if len(contacts) > 0:
                    return [[c[1], c[5], c[9]] for c in contacts]
            else:
                contacts = [c for c in contacts if c[3] != 9 and c[3] != 10]
                if len(contacts) > 0:
                    return [[c[1], c[5], c[9]] for c in contacts]
        return []
    


class BulletFrankaGripper(BulletRobot):
    robot_type = FrankaGripper

    def _set_robot_specifics(self, default_prismatic_value=0.025):
        self.default_prismatic_value = default_prismatic_value

    def _set_gripper_constraint(self):
        pass

    def marionette(self, state, frame="right_gripper"):
        assert isinstance(state, SE3)
        assert frame in ["base_frame", "right_gripper", "panda_grasptarget"]
        # Pose is expressed as a transformation from the desired frame to the world
        # But we need to transform it into the base frame

        # TODO maybe there is some way to cache these transforms from the urdf
        # instead of hardcoding them
        if frame == "right_gripper":
            transform = SE3.from_matrix(
                np.array(
                    [
                        [-0.7071067811865475, 0.7071067811865475, 0, 0],
                        [-0.7071067811865475, -0.7071067811865475, 0, 0],
                        [0, 0, 1, -0.1],
                        [0, 0, 0, 1],
                    ]
                )
            )
            state = state * transform
        elif frame == 'panda_grasptarget':
            transform = SE3.from_matrix(
                np.array(
                    [
                        [0.7071067811865475, 0.7071067811865475, 0, 0],
                        [0.7071067811865475, 0.7071067811865475, 0, 0],
                        [0, 0, 1, -0.105],
                        [0, 0, 0, 1],
                    ]
                )
            )
            state = state * transform

        x, y, z = state.xyz
        p.resetJointState(self.id, 0, x, physicsClientId=self.clid)
        p.resetJointState(self.id, 1, y, physicsClientId=self.clid)
        p.resetJointState(self.id, 2, z, physicsClientId=self.clid)
        p.resetJointStateMultiDof(self.id, 3, state.so3.xyzw, physicsClientId=self.clid)
        p.resetJointState(
            self.id, 5, self.default_prismatic_value, physicsClientId=self.clid
        )
        p.resetJointState(
            self.id, 6, self.default_prismatic_value, physicsClientId=self.clid
        )

    def in_collision(self, obstacles, radius=0.0, check_self=False):
        """
        Checks whether the robot is in collision with the environment

        :return: Boolean
        """
        # Step the simulator (only enough for collision detection)
        if check_self:
            contacts = p.getClosestPoints(
                self.id, self.id, radius, physicsClientId=self.clid
            )
            # Manually filter out fixed connections that shouldn't be considered
            # TODO fix this somehow
            filtered = []
            for c in contacts:
                # A link is always in collision with itself and its neighbors
                if abs(c[3] - c[4]) <= 1:
                    continue
                # panda_link8 just transforms the origin
                if c[3] == 6 and c[4] == 4:
                    continue
                if c[3] == 4 and c[4] == 6:
                    continue
                filtered.append(c)
            if len(filtered) > 0:
                return True

        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getClosestPoints(
                self.id, id, radius, physicsClientId=self.clid
            )
            if len(contacts) > 0:
                return True
        return False


class VisualGripper:
    def __init__(self, clid, **kwargs):
        self.id = self.load(clid, **kwargs)
        self.clid = clid

    def load(self, clid, prismatic=0.025):
        if math.isclose(prismatic, 0.04):
            path = FrankaGripper.fully_open_mesh
        elif math.isclose(prismatic, 0.025):
            path = FrankaGripper.half_open_mesh
        else:
            raise NotImplementedError(
                "Only prismatic values [0.04, 0.025] currently supported for VisualGripper"
            )

        obstacle_visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=path,
            # rgbaColor=[0, 1.0, 0, 1],
            physicsClientId=clid,
        )
        return p.createMultiBody(
            basePosition=[0, 0, 0],  # t.center,
            baseOrientation=[0, 0, 0, 1],  # t.pose.so3.xyzw,
            baseVisualShapeIndex=obstacle_visual_id,
            physicsClientId=clid,
        )

    def marionette(self, pose):
        p.resetBasePositionAndOrientation(
            self.id, pose.xyz, pose.so3.xyzw, physicsClientId=self.clid
        )


class Bullet:
    def __init__(self, gui=False, headless=False):
        assert not (gui and headless), 'GUI cannot be turned on for headless mode'
        self.use_gui = gui
        self.headless = headless
        if self.use_gui:
            self.clid = p.connect(p.GUI)
        else:
            self.clid = p.connect(p.DIRECT)
        self.robots = {}
        self.obstacle_ids = []
        self.poses = []

    def __del__(self):
        p.disconnect(self.clid)

    def load_robot(self, robot_type, hd=False, collision_free=False, **kwargs):
        if robot_type == FrankaRobot:
            robot = BulletFranka(self.clid, hd, **kwargs)
        elif robot_type == FrankaGripper:
            if collision_free:
                robot = VisualGripper(self.clid, **kwargs)
            else:
                robot = BulletFrankaGripper(self.clid, **kwargs)
        self.robots[robot.id] = robot
        return robot

    def in_collision(self, robot, radius=0.0, check_self=False, **kwargs):
        return robot.in_collision(self.obstacle_ids, radius, check_self, **kwargs)
    
    def in_collision_table(self):
        contacts = []
        table_id = self.obstacle_ids[1]
        for id in self.obstacle_ids[2:]:
            _contacts = p.getContactPoints(table_id, id, physicsClientId=self.clid)
            if _contacts:
                contact_force = np.sum([c[9] for c in _contacts])
                contact_pos = np.mean([c[5] for c in _contacts], axis=0)
                contacts.append([id, contact_force, contact_pos])
        return contacts
                
    def save_state(self):
        self.snapshot_id = p.saveState(physicsClientId=self.clid)

    def restore_state(self):
        p.restoreState(self.snapshot_id)


    def visualize_pose(self, pose):
        root_path = Path("/tmp")

        target_ids = []
        x_mesh_path = root_path / "target_x.obj"
        if not x_mesh_path.exists():
            x_transform = np.array(
                [[0, 0, 1, 0.05], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
            )
            x_axis_mesh = trimesh.creation.cylinder(
                radius=0.005,
                height=0.1,
                transform=x_transform,
            )
            with open(x_mesh_path, "w") as f:
                f.write(
                    trimesh.exchange.obj.export_obj(x_axis_mesh, include_color=True)
                )
        obstacle_visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(x_mesh_path),
            physicsClientId=self.clid,
            rgbaColor=[1, 0, 0, 1],
        )
        target_ids.append(
            p.createMultiBody(
                basePosition=pose.xyz,
                baseOrientation=pose.so3.xyzw,
                baseVisualShapeIndex=obstacle_visual_id,
                physicsClientId=self.clid,
            )
        )

        y_mesh_path = root_path / "target_y.obj"
        if not y_mesh_path.exists():
            y_transform = np.array(
                [[0, 1, 0, 0], [0, 0, 1, 0.05], [1, 0, 0, 0], [0, 0, 0, 1]]
            )
            y_axis_mesh = trimesh.creation.cylinder(
                radius=0.005,
                height=0.1,
                transform=y_transform,
            )
            with open(y_mesh_path, "w") as f:
                f.write(
                    trimesh.exchange.obj.export_obj(y_axis_mesh, include_color=True)
                )
        obstacle_visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(y_mesh_path),
            physicsClientId=self.clid,
            rgbaColor=[0, 1, 0, 1],
        )
        target_ids.append(
            p.createMultiBody(
                basePosition=pose.xyz,
                baseOrientation=pose.so3.xyzw,
                baseVisualShapeIndex=obstacle_visual_id,
                physicsClientId=self.clid,
            )
        )
        z_mesh_path = root_path / "target_z.obj"
        if not z_mesh_path.exists():
            z_transform = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.05], [0, 0, 0, 1]]
            )
            z_axis_mesh = trimesh.creation.cylinder(
                radius=0.005, height=0.1, transform=z_transform
            )
            with open(z_mesh_path, "w") as f:
                f.write(
                    trimesh.exchange.obj.export_obj(z_axis_mesh, include_color=True)
                )
        obstacle_visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(z_mesh_path),
            physicsClientId=self.clid,
            rgbaColor=[0, 0, 1, 1],
        )
        target_ids.append(
            p.createMultiBody(
                basePosition=pose.xyz,
                baseOrientation=pose.so3.xyzw,
                baseVisualShapeIndex=obstacle_visual_id,
                physicsClientId=self.clid,
            )
        )
        self.poses.append(target_ids)
        return target_ids

    def set_camera_position(self, yaw, pitch, distance, target):
        p.resetDebugVisualizerCamera(
            distance, yaw, pitch, target, physicsClientId=self.clid
        )

    def set_camera_position_from_matrix(self, pose):
        distance = pose.matrix[2, 3] / pose.matrix[2, 2]
        # distance = 5
        target = tuple((pose.matrix @ np.array([0, 0, -distance, 1]))[:3])

        # Calculations for signed angles come from here https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
        # Have to get the angle between the camera's x axis and the world x axis
        v1 = (pose.matrix @ np.array([1, 0, 0, 1]))[:3] - pose.pos
        v2 = np.array([1, 0, 0])
        vn = np.cross(v1, v2)
        vn = vn / np.linalg.norm(vn)
        yaw = np.degrees(np.arctan2(np.dot(np.cross(v1, v2), vn), np.dot(v1, v2)))

        # Have to get the
        v1 = (pose.matrix @ np.array([0, 1, 0, 1]))[:3] - pose.pos
        v2 = np.array([0, 0, 1])
        vn = np.cross(v1, v2)
        vn = vn / np.linalg.norm(vn)
        pitch = np.degrees(np.arctan2(np.dot(np.cross(v2, v1), vn), np.dot(v1, v2)))
        p.resetDebugVisualizerCamera(
            distance, yaw, pitch, target, physicsClientId=self.clid
        )

    def get_camera_position(self):
        params = p.getDebugVisualizerCamera(physicsClientId=self.clid)
        return {'yaw': params[8],
                'pitch': params[9],
                'distance': params[10],
                'target': params[11]}


    def get_camera_images(
        self,
        extrinsic,
        width=640,
        height=480,
        fx=616.36529541,
        fy=616.20294189,
        cx=310.25881958,
        cy=310.25881958,
        near=0.01,
        far=10,
        scale=True,
    ):
        perspective = np.array([[fx, 0.0, -cx, 0.0],
                                [0.0, fy, -cy, 0.0],
                                [0.0, 0.0, near + far, near * far],
                                [0.0, 0.0, -1.0, 0.0]])
        left, right, bottom, top = 0.0, width, height, 0.0
        ortho = np.diag([2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0])
        ortho[0, 3] = -(right + left) / (right - left)
        ortho[1, 3] = -(top + bottom) / (top - bottom)
        ortho[2, 3] = -(far + near) / (far - near)
        projection_matrix = ortho @ perspective

        view_matrix = copy.deepcopy(extrinsic)
        view_matrix[2, :] *= -1  # flip the Z axis
        view_matrix = view_matrix.flatten(order='F')
        projection_matrix = projection_matrix.flatten(order='F')

        _, _, rgb, depth, seg = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.clid,
        )
        if scale:
            depth = far * near / (far - (far - near) * depth)
        return rgb, depth, seg

    def get_pointcloud_from_camera(
        self,
        extrinsic,
        depth_image,
        segmentation,        
        width=640,
        height=480,
        fx=616.36529541,
        fy=616.20294189,
        cx=310.25881958,
        cy=310.25881958,
        far=10,
        finite_depth=True,
    ):
        # Remove all points that are too far away
        depth_image[depth_image > far] = 0.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        pcs = []
        for i in range(5):
            if i == 0: # all
                _depth_image = copy.deepcopy(depth_image)
                _depth_image[segmentation == -1] = 0.0
                _depth_image[segmentation == 0] = 0.0
            elif i == 1: # robot
                _depth_image = copy.deepcopy(depth_image)
                _depth_image[segmentation != 1] = 0.0
            elif i == 2: # mount table
                _depth_image = copy.deepcopy(depth_image)
                _depth_image[segmentation != 2] = 0.0
            elif i == 3: # front object
                _depth_image = copy.deepcopy(depth_image)
                _depth_image[segmentation != 3] = 0.0
            elif i == 4: # object
                _depth_image = copy.deepcopy(depth_image)
                _depth_image[segmentation < 4] = 0.0
                                
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            ones = np.ones((height, width))
            image_points = np.stack((x, y, ones), axis=2).reshape(width * height, 3).T
            backprojected = np.linalg.inv(K) @ image_points
            pc = np.multiply(np.tile(_depth_image.reshape(1, width * height), (3, 1)),
                            backprojected).T
            if finite_depth:
                pc = pc[np.isfinite(pc[:, 0]), :]
            pc = transform_pointcloud(pc, np.linalg.inv(extrinsic), in_place=False)
            
            pcs.append(pc)
        pcs = np.array(pcs)
        return pcs

    '''
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)
o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)])
    '''
    
    
    def load_mesh(self, visual_mesh_path, collision_mesh_path=None, color=None):
        if collision_mesh_path is None:
            collision_mesh_path = visual_mesh_path
        if color is None:
            color = [0.85882353, 0.14117647, 0.60392157, 1]
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=visual_mesh_path,
            rgbaColor=color,
            physicsClientId=self.clid,
        )
        collision_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=collision_mesh_path,
            physicsClientId=self.clid,
        )
        obstacle_id = p.createMultiBody(
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            baseVisualShapeIndex=visual_id,
            baseCollisionShapeIndex=collision_id,
            physicsClientId=self.clid,
        )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id


    def load_cuboid_cylinder(self, primitive, color, mass, lateral_friction, rolling_friction, visual_only=False):
        assert isinstance(primitive, Cuboid) or isinstance(primitive, Cylinder)
        if color is None:
            color = [0.85882353, 0.14117647, 0.60392157, 1]
        assert not primitive.is_zero_volume(), 'Cannot load zero volume primitive'
        kwargs = {}
        if self.use_gui or self.headless:
            if isinstance(primitive, Cuboid):
                obstacle_visual_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                         halfExtents=primitive.half_extents.tolist(),
                                                         rgbaColor=color,
                                                         physicsClientId=self.clid)
            else:
                obstacle_visual_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                                         radius=primitive.radius,
                                                         length=primitive.height,
                                                         rgbaColor=color,
                                                         physicsClientId=self.clid)
            kwargs['baseVisualShapeIndex'] = obstacle_visual_id
        if not visual_only:
            if isinstance(primitive, Cuboid):
                obstacle_collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                               halfExtents=primitive.half_extents.tolist(),
                                                               physicsClientId=self.clid)
            else:
                obstacle_collision_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                               radius=primitive.radius,
                                                               height=primitive.height,
                                                               physicsClientId=self.clid)
            kwargs['baseCollisionShapeIndex'] = obstacle_collision_id
        
        obstacle_id = p.createMultiBody(baseMass=mass,
                                        basePosition=primitive.center.tolist(),
                                        baseOrientation=primitive.pose.so3.xyzw,
                                        physicsClientId=self.clid,
                                        **kwargs)
        p.changeDynamics(obstacle_id,
                         -1,
                         lateralFriction=lateral_friction,
                         rollingFriction=rolling_friction,
                         physicsClientId=self.clid)
        
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def load_primitives(self, primitives, color=None, visual_only=False):
        ids = []
        for prim in primitives:
            if prim.primitive.is_zero_volume():
                continue
            if isinstance(prim.primitive, Cuboid) or isinstance(prim.primitive, Cylinder):
                ids.append(self.load_cuboid_cylinder(prim.primitive, prim.color, prim.mass, prim.lateral_friction, prim.rolling_friction, visual_only))
        return ids


    def load_urdf_obstacle(self, path, pose=None):
        if pose is not None:
            obstacle_id = p.loadURDF(
                str(path),
                basePosition=pose.xyz,
                baseOrientation=pose.so3.xyzw,
                useFixedBase=True,
                physicsClientId=self.clid,
            )
        else:
            obstacle_id = p.loadURDF(
                str(path),
                useFixedBase=True,
                physicsClientId=self.clid,
            )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def clear_obstacle(self, id):
        """
        Removes a specific obstacle from the environment

        :param id: Bullet id of obstacle to remove
        """
        if id is not None:
            p.removeBody(id, physicsClientId=self.clid)
            self.obstacle_ids = [x for x in self.obstacle_ids if x != id]

    def clear_all_obstacles(self):
        """
        Removes all obstacles from bullet environment
        """
        for id in self.obstacle_ids:
            if id is not None:
                p.removeBody(id, physicsClientId=self.clid)
        self.obstacle_ids = []

    def clear_all_poses(self):
        for pose in self.poses:
            for id in pose:
                p.removeBody(id, physicsClientId=self.clid)
        self.poses = []


class BulletController(Bullet):
    def __init__(self, gui=False, hz=12):
        super().__init__(gui, not gui)
        self.gui = gui
        self.dt = 1 / hz
        self.solver_iterations = 150
        p.setPhysicsEngineParameter(
            fixedTimeStep=1 / hz,
            numSolverIterations=self.solver_iterations,
            # numSubSteps=substeps,
            # deterministicOverlappingPairs=1,
            physicsClientId=self.clid,
        )
        p.setGravity(0.0, 0.0, -9.81)
        self.sim_time = 0.0

        uid = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.clid)
        p.createMultiBody(baseMass=0.0,
                          baseCollisionShapeIndex=uid,
                          baseVisualShapeIndex=uid,
                          basePosition=[0.0, 0.0, -0.02],
                          physicsClientId=self.clid)

    def step(self):
        p.stepSimulation(physicsClientId=self.clid)
        self.sim_time += self.dt
