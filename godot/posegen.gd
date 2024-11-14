extends Node3D

@export var overall_scale = 1.0
@export var overall_height = 1.0
@export var draw_landmarks : bool = false
@export var draw_annotation : bool = true
@export var marker_size = 0.025

var worldPoints := PackedVector3Array()
var localPelvis := Vector3()
var localChest := Vector3()
var ik3dSpine := Node3D.new()
var ik3dNeck := Node3D.new()
var ik3dRightLeg := Node3D.new()
var ik3dRightFoot := Node3D.new()
var ik3dRightArm := Node3D.new()
var ik3dLeftLeg := Node3D.new()
var ik3dLeftFoot := Node3D.new()
var ik3dLeftArm := Node3D.new()
var skel3d := Skeleton3D.new()

var pelvisIndex := int()
var chestIndex := int()
var finger_indices = []
var markers = []

var annotation := Node2D.new()
var server := UDPServer.new()

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	#server.listen(4242)
	_init_iks()
	_init_gui_elems()

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta: float) -> void:
	if len(worldPoints):
		_pose_iks()
		_draw_gui_elems()

func _init_iks():
	ik3dSpine = get_node("Mannequiney/root/Skeleton3D/IK3DSpine")
	ik3dNeck = get_node("Mannequiney/root/Skeleton3D/IK3DNeck")
	ik3dRightLeg = get_node("Mannequiney/root/Skeleton3D/IK3DRightLeg")
	ik3dRightFoot = get_node("Mannequiney/root/Skeleton3D/IK3DRightFoot")
	ik3dRightArm = get_node("Mannequiney/root/Skeleton3D/IK3DRightArm")
	ik3dLeftLeg = get_node("Mannequiney/root/Skeleton3D/IK3DLeftLeg")
	ik3dLeftFoot = get_node("Mannequiney/root/Skeleton3D/IK3DLeftFoot")
	ik3dLeftArm = get_node("Mannequiney/root/Skeleton3D/IK3DLeftArm")
	# Compute pelvis & chest offsets local to the transform derived from 4 limbs
	skel3d = get_node("Mannequiney/root/Skeleton3D")
	var lleg = skel3d.get_bone_global_rest(skel3d.find_bone('thigh.l')).origin
	var rleg = skel3d.get_bone_global_rest(skel3d.find_bone('thigh.r')).origin
	var larm = skel3d.get_bone_global_rest(skel3d.find_bone('upperarm.l')).origin
	var rarm = skel3d.get_bone_global_rest(skel3d.find_bone('upperarm.r')).origin
	pelvisIndex = skel3d.find_bone('pelvis')
	var pelvis = skel3d.get_bone_global_rest(pelvisIndex).origin
	localPelvis = _derive_pelvis_xform(larm, rarm, lleg, rleg).inverse() * pelvis
	chestIndex = skel3d.find_bone('spine_02')
	var chest = skel3d.get_bone_global_rest(chestIndex).origin
	localChest = _derive_chest_xform(larm, rarm, lleg, rleg).inverse() * chest
	# Store finger indices so we can adjust the bones later
	for finger in ['index', 'middle', 'ring', 'pinky']:
		for side in ['r','l']:
			for i in range(1, 4):
				var bone_index = skel3d.find_bone(finger + "_0" + str(i) + "." + side)
				if bone_index >= 0:
					finger_indices.append(bone_index)

func _pose_iks():
	var P = PackedVector3Array()
	for p in worldPoints:
		P.append(overall_scale * p + Vector3(0,overall_height,0))
	# Derive pelvis & chest from the limb locations
	var larm = P[11]
	var rarm = P[12]
	var lleg = P[23]
	var rleg = P[24]
	var pelvisxf = _derive_pelvis_xform(larm, rarm, lleg, rleg)
	var pelvispos = pelvisxf * localPelvis
	skel3d.set_bone_pose(pelvisIndex, Transform3D(pelvisxf.basis, pelvispos))
	var chestxf = _derive_chest_xform(larm, rarm, lleg, rleg)
	var chestpos = chestxf * localChest
	var headxf = _derive_head_xform(P[7], P[8], (P[0]+P[1]+P[4]) / 3)
	# Pose torso, chest, neck, & head
	ik3dSpine.start()
	ik3dSpine.target = Transform3D(chestxf.basis, chestpos)
	ik3dNeck.start()
	ik3dNeck.target = headxf
	# Pose right limbs
	ik3dRightLeg.start()
	ik3dRightLeg.target.origin = P[28]
	ik3dRightLeg.magnet = P[26]
	ik3dRightFoot.start()
	ik3dRightFoot.target = _derive_foot_xform(P[28], P[30], P[32])
	ik3dRightArm.start()
	ik3dRightArm.target = _derive_hand_xform(P[16],P[20], P[18], true)
	ik3dRightArm.magnet = P[14]
	# Pose left limbs
	ik3dLeftLeg.start()
	ik3dLeftLeg.target.origin = P[27]
	ik3dLeftLeg.magnet = P[25]
	ik3dLeftFoot.start()
	ik3dLeftFoot.target = _derive_foot_xform(P[27], P[29], P[31])
	ik3dLeftFoot.magnet = P[31]
	ik3dLeftArm.start()
	ik3dLeftArm.target = _derive_hand_xform(P[15], P[19], P[17], false)
	ik3dLeftArm.magnet = P[13]
	# Adjust the fingers to make the palm open slightly
	var rot = Quaternion(Vector3(1,0,0), -0.5)
	for findex in finger_indices:
		skel3d.set_bone_pose_rotation(findex, rot)

func _derive_pelvis_xform(larm: Vector3, rarm: Vector3, lleg: Vector3, rleg: Vector3):
	var centre = (larm + rarm + rleg + lleg) / 4
	var origin = (rleg + lleg) / 2
	var bx = (lleg - origin).normalized()
	var by = (centre - origin).normalized()
	var bz = bx.cross(by)
	by = bz.cross(bx)
	return Transform3D(bx, by, bz, origin)

func _derive_chest_xform(larm: Vector3, rarm: Vector3, lleg: Vector3, rleg: Vector3):
	var centre = (larm + rarm + rleg + lleg) / 4
	var origin = (larm + rarm) / 2
	var bx = (larm - origin).normalized()
	var by = -1 * (centre - origin).normalized()
	var bz = bx.cross(by)
	by = bz.cross(bx)
	return Transform3D(bx, by, bz, origin)

func _derive_head_xform(lear: Vector3, rear: Vector3, nose: Vector3):
	var origin = (lear + rear) / 2
	var bx = (lear - origin).normalized()
	var bz = (nose - origin).normalized()
	var by = bz.cross(bx)
	bz = bx.cross(by)
	return Transform3D(bx, by, bz, origin)

func _derive_hand_xform(wrist: Vector3, index: Vector3, pinky: Vector3, flip: bool):
	var centre = (wrist + index + pinky) / 3
	var by = (centre - wrist).normalized()
	var bz = (index - wrist).normalized().cross((pinky - wrist).normalized())
	var bx = by.cross(bz)
	bx *= -1 if flip else 1
	bz *= -1 if flip else 1
	return Transform3D(bx, by, bz, wrist)

func _derive_foot_xform(ankle: Vector3, heel: Vector3, toe: Vector3):
	var centre = (toe + heel) / 2
	var by = (toe - heel).normalized()
	var bx = (ankle - heel).normalized().cross(by)
	var bz = bx.cross(by)
	return Transform3D(bx, by, bz, centre)

func _init_gui_elems():
	for i in range(33):
		var sphere = CSGSphere3D.new()
		sphere.visible = false
		sphere.physics_interpolation_mode = Node.PHYSICS_INTERPOLATION_MODE_ON
		add_child(sphere)
		markers.append(sphere)
	annotation = get_node('Gui/Sprite2D')

func _draw_gui_elems():
	if draw_landmarks:
		for i in range(len(worldPoints)):
			markers[i].position = overall_scale * worldPoints[i] + Vector3(0,overall_height,0)
			markers[i].radius = marker_size
			markers[i].visible = true
	else:
		for x in markers:
			x.visible = false
	annotation.visible = draw_annotation
