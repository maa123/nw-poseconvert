'use strict';
const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');
const Canvas = require('canvas');
const Image = Canvas.Image;
const fs = require('fs');
const path = require('path');

const frames_dir = './tmp_frames';

const getFileList = async(fpath) => new Promise((s, r) => {
	fs.readdir(fpath, {withFileTypes: true}, (err, dirents) => {
		if (err) {
			r(err);
			return;
		}
		const list = [];
		for (let i = 0; i < dirents.length; i++) {
			if(!dirents[i].isDirectory()){
				list.push(path.join(fpath, dirents[i].name));
			}
		}
		s(list);
	});
});

const getPos = (keypoint) => {
	return {x: keypoint.position.x, y: keypoint.position.y};
}

const drawLines = (posekey, lines, ctx) => {
	for (let i = 0; i < lines.length; i++) {
		const p1 = getPos(posekey[lines[i][0]]);
		const p2 = getPos(posekey[lines[i][1]]);
		ctx.beginPath();
		ctx.moveTo(p1.x, p1.y);
		ctx.lineTo(p2.x, p2.y);
		ctx.closePath();
		ctx.stroke();
	}
}

const drawPose = (pose, ctx) => {
    ctx.fillStyle = '#f00';
	const posekey = pose.keypoints;
	for (let i = 0; i < posekey.length; i++) {
		const ps = posekey[i].position;
		ctx.beginPath();
		ctx.arc(ps.x, ps.y, 5, 0, Math.PI*2);
		ctx.fill();
	}

	ctx.strokeStyle = '#0f0';
	ctx.lineWidth = 2;

	const lines = [[5,6],[5,7],[6,8],[7,9],[8,10],[5,11],[6,12],[11,12],[11,13],[12,14],[13,15],[14,16]];
	drawLines(posekey, lines, ctx);
}

const PoseDetect = async(net, fpath) => {

	const img = new Image();
	img.src = fpath;

	const cv = Canvas.createCanvas(img.width, img.height);
	const ctx = cv.getContext('2d');
	ctx.drawImage(img, 0, 0);

	const poses = await net.estimateMultiplePoses(cv, {
	  flipHorizontal: false,
	  maxDetections: 6,
	  scoreThreshold: 0.5,
	  nmsRadius: 20
	});
	for (let i = 0; i < poses.length; i++) {
		drawPose(poses[i], ctx);
	}
	fs.writeFileSync(fpath, cv.toBuffer());
	return;
};

(async()=>{

const net = await posenet.load();

const list = await getFileList(frames_dir);

for (let i = 0; i < list.length; i++) {
    process.stdout.write("\r");
	await processPosedImg(net, list[i]);
	process.stdout.write(`${i}/${list.length}`);
}
process.stdout.write("\n");
})();