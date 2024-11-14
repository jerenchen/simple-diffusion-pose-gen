extends LineEdit


var main := Node3D.new()
var host := "127.0.0.1"
var port := 42442

func _ready() -> void:
	main = get_node('../..')

func _on_text_submitted(new_text: String) -> void:
	var socket := PacketPeerUDP.new()
	socket.set_dest_address(host, port)
	var data = {'prompt':new_text}
	socket.put_packet(JSON.stringify(data).to_ascii_buffer())
	
	while socket.wait() == OK:
		var json = JSON.new()
		json.parse(socket.get_packet().get_string_from_ascii())
		if "landmarks" in json.data:
			main.worldPoints.clear()
			for p in json.data["landmarks"]:
				main.worldPoints.append(Vector3(p[0],p[1],p[2]))
			if "image" in json.data:
				var image = Image.load_from_file(json.data['image'])
				main.annotation.texture = ImageTexture.create_from_image(image)
			return
