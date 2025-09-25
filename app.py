# backend/app.py

import os
import subprocess
import secrets
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, jwt_required, JWTManager, get_jwt_identity
from datetime import datetime, timezone
import time
import base64
from cryptography.fernet import Fernet, InvalidToken
import json

# --- 기존 유틸리티 함수 (변경 없음) ---
def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary_string):
    chars = [binary_string[i:i+8] for i in range(0, len(binary_string), 8)]
    text = []
    for char_bin in chars:
        if len(char_bin) == 8:
            try:
                text.append(chr(int(char_bin, 2)))
            except ValueError:
                pass
    return "".join(text)

# --- Flask 애플리케이션 설정 ---
app = Flask(__name__)
# [수정] CORS 설정을 환경 변수에서 프론트엔드 URL을 읽어오도록 변경
# 개발 환경에서는 localhost, 배포 환경에서는 Vercel URL을 허용합니다.
frontend_url = os.environ.get('FRONTEND_URL')
allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
if frontend_url:
    # Vercel 주소 뒤에 붙는 / 를 제거하여 정확한 Origin과 일치시킵니다.
    allowed_origins.append(frontend_url.rstrip('/'))

CORS(app, origins=allowed_origins, supports_credentials=True, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# [추가] 데이터베이스 및 인증 설정
# [수정] 서버가 재시작되어도 JWT 서명이 일관되도록 고정된 시크릿 키를 사용합니다.
# 실제 운영 환경에서는 이 값을 환경 변수에서 불러와야 합니다.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-super-secret-and-static-key-for-development-only')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'a-super-secret-and-static-key-for-development-only')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# [추가] 암호화 설정
# Fernet 키는 32바이트여야 합니다. SECRET_KEY를 기반으로 일관된 키를 생성합니다.
# 키가 32바이트보다 길 경우 잘라내고, 짧을 경우 패딩합니다.
key_32_bytes = app.config['SECRET_KEY'].encode().ljust(32)[:32]
fernet_key = base64.urlsafe_b64encode(key_32_bytes)
cipher_suite = Fernet(fernet_key)

# [추가] 사용자 모델 정의
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user') # [추가] 사용자 역할 (user, admin)

    def __repr__(self):
        return f"User('{self.username}')"

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'role': self.role
        }

# [추가] 비디오 모델 정의
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=True)
    original_filename = db.Column(db.String(100), nullable=False)
    master_filename = db.Column(db.String(100), unique=True, nullable=False)
    playback_filename = db.Column(db.String(100), unique=True, nullable=False)
    thumbnail_filename = db.Column(db.String(100), nullable=True) # New field for thumbnail
    upload_timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('videos', lazy=True, cascade="all, delete-orphan"))

    def to_dict(self):
        thumbnail_filename = getattr(self, 'thumbnail_filename', None)
        return {
            'id': self.id,
            'title': self.title or self.original_filename,
            'original_filename': self.original_filename,
            'master_filename': self.master_filename,
            'playback_filename': self.playback_filename,
            'thumbnail_url': f'/outputs/{thumbnail_filename}' if thumbnail_filename else None,
            'upload_timestamp': self.upload_timestamp.isoformat(),
            'user': {'username': self.user.username} # [추가] 비디오를 업로드한 사용자 정보
        }

# [추가] 댓글 모델 정의
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('comments', lazy=True, cascade="all, delete-orphan"))
    video = db.relationship('Video', backref=db.backref('comments', lazy=True, cascade="all, delete-orphan"))

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'timestamp': self.timestamp.isoformat(),
            'user': self.user.to_dict()
        }
# [추가] 앱 시작 시 데이터베이스 테이블 자동 생성
# 이 코드는 매번 서버를 시작할 때마다 데이터베이스와 테이블이 존재하는지 확인하고,
# 없으면 자동으로 생성해줍니다. 수동으로 db.create_all()을 실행할 필요가 없습니다.
with app.app_context():
    db.create_all()

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 폴더가 없으면 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --- 기존 함수를 API에 맞게 수정 ---
def embed_watermark(input_path, output_path, watermark_text):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, f"Error: '{input_path}' 동영상을 열 수 없습니다."

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # [수정] 데이터 손실이 없는 무손실 코덱(FFV1)으로 변경합니다.
    # 이렇게 하면 압축 과정에서 워터마크가 손상되지 않습니다.
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    watermark_binary = text_to_binary(watermark_text + '$$END$$')
    watermark_len = len(watermark_binary)

    # [수정] 워터마크를 저장할 수 있는 공간을 프레임의 너비(width)에서 프레임 전체 픽셀(width*height)로 확장합니다.
    available_bits = width * height
    if watermark_len > available_bits:
        cap.release()
        out.release()
        return False, f"Error: 워터마크가 너무 깁니다. (최대 {available_bits} 비트, 현재 {watermark_len} 비트)"

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 첫 번째 프레임에만 워터마크를 삽입합니다.
        if frame_count == 0:
            # [수정] 1차원 인덱스를 2차원 (행, 열) 좌표로 변환하여 프레임 전체에 걸쳐 워터마크를 삽입합니다.
            for i in range(watermark_len):
                row = i // width
                col = i % width
                pixel = frame[row, col]
                blue_val = pixel[0]
                watermark_bit = int(watermark_binary[i])
                # [수정] 무손실 코덱을 사용하므로 가장 간단한 최하위 비트(LSB) 방식으로 되돌립니다.
                new_blue_val = (blue_val & 0b11111110) | watermark_bit
                frame[row, col, 0] = new_blue_val
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    return True, "워터마크 삽입 완료"

def extract_watermark(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return f"Error: '{input_path}' 동영상을 열 수 없습니다."

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "Error: 동영상에서 프레임을 읽을 수 없습니다."

    height, width, _ = frame.shape
    extracted_binary = ""
    terminator_binary = text_to_binary("$$END$$")

    # [수정] 프레임의 첫 번째 행만 읽던 것에서 프레임 전체를 순회하며 워터마크 비트를 추출하도록 변경합니다.
    for i in range(height * width):
        row = i // width
        col = i % width
        pixel = frame[row, col]
        blue_val = pixel[0]
        # [수정] 삽입 방식과 동일하게 최하위 비트(LSB)에서 데이터를 추출합니다.
        lsb = blue_val & 1
        extracted_binary += str(lsb)
        # [추가] 종료 문자를 찾으면 더 이상 읽지 않고 중단하여 효율성을 증대시킵니다.
        if extracted_binary.endswith(terminator_binary):
            break

    extracted_text = binary_to_text(extracted_binary)
    terminator = "$$END$$"
    if terminator in extracted_text:
        watermark = extracted_text.split(terminator)[0]
    else:
        watermark = "워터마크를 찾지 못했거나 데이터가 손상되었습니다."
    
    cap.release()
    return watermark

# --- [추가] 인증 API 엔드포인트 ---
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': '사용자 이름과 비밀번호를 모두 입력해주세요.'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'error': '이미 존재하는 사용자 이름입니다.'}), 409

    # [추가] 첫 번째 가입자를 admin으로 자동 지정
    role = 'user'
    if User.query.first() is None:
        app.logger.info(f"First user '{username}' is being set as admin.")
        role = 'admin'

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(username=username, password=hashed_password, role=role)
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': '회원가입 성공!'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()

    if user and bcrypt.check_password_hash(user.password, password):
        # [수정] JWT 토큰에 역할(role) 정보도 포함시킴
        identity_data = {'username': user.username, 'id': user.id, 'role': user.role}
        access_token = create_access_token(identity=json.dumps(identity_data))
        return jsonify(access_token=access_token)

    return jsonify({'error': '사용자 이름 또는 비밀번호가 일치하지 않습니다.'}), 401

# [추가] 프로필 관리 API - 비밀번호 변경
@app.route('/api/profile/change-password', methods=['POST'])
@jwt_required()
def change_password():
    identity_json = get_jwt_identity()
    current_user_data = json.loads(identity_json)
    user_id = current_user_data['id']
    
    data = request.get_json()
    current_password = data.get('current_password')
    new_password = data.get('new_password')

    if not current_password or not new_password:
        return jsonify({'error': '현재 비밀번호와 새 비밀번호를 모두 입력해주세요.'}), 400

    user = User.query.get(user_id)

    if not user or not bcrypt.check_password_hash(user.password, current_password):
        return jsonify({'error': '현재 비밀번호가 일치하지 않습니다.'}), 401

    user.password = bcrypt.generate_password_hash(new_password).decode('utf-8')
    db.session.commit()

    return jsonify({'message': '비밀번호가 성공적으로 변경되었습니다.'})

# --- API 엔드포인트 ---
@app.route('/api/embed', methods=['POST'])
@jwt_required()
def embed_route():
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)
    original_watermark = f"user_id:{current_user['id']},username:{current_user['username']}"
    app.logger.info(f"Watermark embed request by {original_watermark}")

    if 'video' not in request.files:
        return jsonify({'error': '비디오 파일이 없습니다.'}), 400

    video_file = request.files['video']

    # [추가] 폼 데이터에서 제목 가져오기 및 유효성 검사
    title = request.form.get('title')
    if not title or not title.strip():
        return jsonify({'error': '영상 제목을 입력해주세요.'}), 400

    # [추가] 썸네일 데이터 가져오기 및 저장
    thumbnail_data_url = request.form.get('thumbnail')
    thumbnail_filename = None
    if thumbnail_data_url:
        try:
            # Data URL (e.g., data:image/jpeg;base64,...)
            header, base64_encoded_data = thumbnail_data_url.split(',', 1)
            decoded_thumbnail = base64.b64decode(base64_encoded_data)
            
            import re
            match = re.search(r'data:image/(\w+);base64', header)
            ext = match.group(1) if match else 'jpeg' # Default to jpeg if format not found
            
            thumbnail_filename = f"thumbnail_{base_filename}.{ext}"
            thumbnail_path = os.path.join(app.config['OUTPUT_FOLDER'], thumbnail_filename)
            with open(thumbnail_path, 'wb') as f:
                f.write(decoded_thumbnail)
        except Exception as e:
            app.logger.error(f"썸네일 저장 중 오류 발생: {e}")
            # Continue without thumbnail if saving fails
            thumbnail_filename = None

    # [추가] 워터마크 암호화
    encrypted_watermark_bytes = cipher_suite.encrypt(original_watermark.encode())
    # 암호화된 바이트를 URL-safe Base64 문자열로 변환하여 LSB 삽입에 사용
    embeddable_watermark = base64.urlsafe_b64encode(encrypted_watermark_bytes).decode('utf-8')

    # 고유한 파일명 생성
    base_filename = f"{int(time.time())}_{video_file.filename.rsplit('.', 1)[0]}"
    filename = f"{base_filename}.{video_file.filename.rsplit('.', 1)[-1]}"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(input_path)

    # [추가] 워터마크 중복 삽입 방지
    try:
        extracted_b64_watermark = extract_watermark(input_path)
        encrypted_watermark_bytes = base64.urlsafe_b64decode(extracted_b64_watermark.encode('utf-8'))
        cipher_suite.decrypt(encrypted_watermark_bytes)
        # 복호화에 성공하면 이미 워터마크가 있는 영상으로 간주
        return jsonify({'error': '이미 워터마크가 삽입된 영상입니다. 중복 삽입은 불가능합니다.'}), 409
    except (InvalidToken, ValueError, TypeError):
        # 복호화에 실패하면 워터마크가 없는 영상으로 간주하고 계속 진행
        pass

    # [수정] 코덱에 맞춰 출력 파일 확장자를 .mkv로 변경합니다.
    master_filename = f"watermarked_{base_filename}.mkv"
    master_path = os.path.join(app.config['OUTPUT_FOLDER'], master_filename)

    success, message = embed_watermark(input_path, master_path, embeddable_watermark)

    if not success:
        return jsonify({'error': message}), 500

    # [추가] 재생 가능한 MP4 파일 생성
    playback_filename = f"playback_{base_filename}.mp4"
    playback_path = os.path.join(app.config['OUTPUT_FOLDER'], playback_filename)
    try:
        # FFmpeg를 사용하여 .mkv를 .mp4로 변환
        subprocess.run(
            ['ffmpeg', '-i', master_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', '-y', playback_path],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
    except subprocess.CalledProcessError as e:
        app.logger.error(f"FFmpeg error: {e.stderr}")
        return jsonify({'error': f'재생 가능한 비디오 변환 실패: {e.stderr}'}), 500
    except FileNotFoundError:
        app.logger.error("FFmpeg not found. Please install FFmpeg and ensure it's in the system's PATH.")
        return jsonify({'error': '서버 오류: FFmpeg가 설치되어 있지 않습니다.'}), 500

    # [추가] 데이터베이스에 비디오 정보 저장
    new_video = Video(
        title=title,
        original_filename=video_file.filename,
        master_filename=master_filename,
        playback_filename=playback_filename,
        thumbnail_filename=thumbnail_filename, # Save thumbnail filename
        user_id=current_user['id']
    )
    db.session.add(new_video)
    db.session.commit()

    # [추가] 처리 완료된 원본 업로드 파일 삭제
    try:
        if os.path.exists(input_path):
            os.remove(input_path)
    except Exception as e:
        # 파일 삭제 실패가 전체 프로세스를 중단시키지 않도록 로깅만 합니다.
        app.logger.error(f"원본 업로드 파일 삭제 실패 ({input_path}): {e}")

    return jsonify({'message': message, 'playback_filename': playback_filename, 'master_filename': master_filename, 'video_id': new_video.id})

@app.route('/api/videos/<int:video_id>', methods=['GET'])
def get_video(video_id):
    # 데이터베이스에서 비디오 ID로 비디오를 찾습니다.
    video = Video.query.get(video_id)
    if not video:
        # 비디오가 없으면 404 에러를 반환합니다.
        return jsonify({"error": "비디오를 찾을 수 없습니다."}), 404
    return jsonify(video.to_dict())

# [추가] 댓글 불러오기 API
@app.route('/api/videos/<int:video_id>/comments', methods=['GET'])
def get_comments(video_id):
    video = Video.query.get_or_404(video_id)
    comments = Comment.query.filter_by(video_id=video.id).order_by(Comment.timestamp.asc()).all()
    return jsonify([comment.to_dict() for comment in comments])

# [추가] 댓글 작성 API
@app.route('/api/videos/<int:video_id>/comments', methods=['POST'])
@jwt_required()
def post_comment(video_id):
    identity_json = get_jwt_identity()
    current_user_data = json.loads(identity_json)
    user_id = current_user_data['id']

    video = Video.query.get_or_404(video_id)
    data = request.get_json()
    text = data.get('text')

    if not text or not text.strip():
        return jsonify({'error': '댓글 내용이 비어있습니다.'}), 400

    comment = Comment(text=text, user_id=user_id, video_id=video.id)
    db.session.add(comment)
    db.session.commit()

    return jsonify(comment.to_dict()), 201

@app.route('/api/my-videos', methods=['GET'])
@jwt_required()
def my_videos():
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)
    page = request.args.get('page', 1, type=int)
    per_page = 8 # 한 페이지에 8개의 비디오를 표시

    pagination = Video.query.filter_by(user_id=current_user['id']).order_by(Video.upload_timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    videos = [video.to_dict() for video in pagination.items]

    return jsonify({
        'videos': videos,
        'total_pages': pagination.pages,
        'current_page': pagination.page,
        'total_videos': pagination.total
    })

# [추가] 사용자 본인 비디오 삭제 API
@app.route('/api/videos/<int:video_id>', methods=['DELETE'])
@jwt_required()
def delete_own_video(video_id):
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)
    user_id = current_user['id']

    video_to_delete = Video.query.get(video_id)

    if not video_to_delete:
        return jsonify({"error": "비디오를 찾을 수 없습니다."}), 404

    # 본인 소유의 비디오인지 확인
    if video_to_delete.user_id != user_id:
        return jsonify({"error": "이 비디오를 삭제할 권한이 없습니다."}), 403

    # 파일 시스템에서 실제 파일 삭제
    try:
        master_path = os.path.join(app.config['OUTPUT_FOLDER'], video_to_delete.master_filename)
        playback_path = os.path.join(app.config['OUTPUT_FOLDER'], video_to_delete.playback_filename)
        
        if os.path.exists(master_path):
            os.remove(master_path)
        if os.path.exists(playback_path):
            os.remove(playback_path)
    except Exception as e:
        app.logger.error(f"파일 삭제 중 오류 발생 (video_id: {video_id}): {e}")
        return jsonify({"error": "비디오 파일 삭제 중 서버에서 오류가 발생했습니다."}), 500

    db.session.delete(video_to_delete)
    db.session.commit()
    return jsonify({"message": "비디오가 성공적으로 삭제되었습니다."})

# [추가] 공개 비디오 목록 API
@app.route('/api/videos/public', methods=['GET'])
def public_videos():
    videos = Video.query.order_by(Video.upload_timestamp.desc()).all()
    return jsonify([video.to_dict() for video in videos])

# [추가] 비디오 워터마크 검증 API
@app.route('/api/videos/<int:video_id>/verify', methods=['GET'])
@jwt_required()
def verify_video_watermark(video_id):
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)
    user_id = current_user['id']

    video = Video.query.get(video_id)

    if not video:
        return jsonify({'error': '비디오를 찾을 수 없습니다.'}), 404

    # 관리자가 아닌 경우, 본인 소유의 비디오만 검증 가능
    if video.user_id != user_id and current_user.get('role') != 'admin':
        return jsonify({'error': '권한이 없습니다.'}), 403

    master_path = os.path.join(app.config['OUTPUT_FOLDER'], video.master_filename)
    if not os.path.exists(master_path):
        return jsonify({'error': '서버에서 원본 파일을 찾을 수 없습니다.'}), 500

    extracted_b64_watermark = extract_watermark(master_path)

    if "Error:" in extracted_b64_watermark or "워터마크를 찾지 못했" in extracted_b64_watermark:
        return jsonify({'watermark': "워터마크를 찾을 수 없습니다."})

    try:
        encrypted_watermark_bytes = base64.urlsafe_b64decode(extracted_b64_watermark.encode('utf-8'))
        decrypted_watermark = cipher_suite.decrypt(encrypted_watermark_bytes).decode('utf-8')
        return jsonify({'watermark': decrypted_watermark})
    except (InvalidToken, ValueError, TypeError) as e:
        app.logger.error(f"Watermark decryption failed for extracted data: {e}")
        return jsonify({'watermark': "워터마크 복호화에 실패했습니다. 데이터가 손상되었거나 유효하지 않습니다."})

# [추가] 관리자 전용 API - 모든 비디오 목록 조회
@app.route('/api/admin/all-videos', methods=['GET'])
@jwt_required()
def all_videos():
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)

    # 관리자 역할인지 확인
    if current_user.get('role') != 'admin':
        return jsonify({"msg": "관리자 권한이 필요합니다."}), 403

    all_vids = Video.query.order_by(Video.upload_timestamp.desc()).all()
    return jsonify([video.to_dict() for video in all_vids])

# [추가] 관리자 전용 API - 모든 사용자 목록 조회
@app.route('/api/admin/all-users', methods=['GET'])
@jwt_required()
def all_users():
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)

    if current_user.get('role') != 'admin':
        return jsonify({"msg": "관리자 권한이 필요합니다."}), 403

    users = User.query.order_by(User.id.asc()).all()
    return jsonify([user.to_dict() for user in users])

# [추가] 관리자 전용 API - 비디오 삭제
@app.route('/api/admin/videos/<int:video_id>', methods=['DELETE'])
@jwt_required()
def delete_video(video_id):
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)

    # 관리자 역할인지 확인
    if current_user.get('role') != 'admin':
        return jsonify({"msg": "관리자 권한이 필요합니다."}), 403

    video_to_delete = Video.query.get(video_id)

    if not video_to_delete:
        return jsonify({"error": "비디오를 찾을 수 없습니다."}), 404

    # 파일 시스템에서 실제 파일 삭제
    try:
        master_path = os.path.join(app.config['OUTPUT_FOLDER'], video_to_delete.master_filename)
        playback_path = os.path.join(app.config['OUTPUT_FOLDER'], video_to_delete.playback_filename)
        
        if os.path.exists(master_path):
            os.remove(master_path)
        if os.path.exists(playback_path):
            os.remove(playback_path)
    except Exception as e:
        app.logger.error(f"파일 삭제 중 오류 발생 (video_id: {video_id}): {e}")
        # 파일 삭제에 실패하더라도 DB 삭제는 계속 진행할 수 있지만, 여기서는 오류를 반환하여 문제를 알립니다.
        return jsonify({"error": "비디오 파일 삭제 중 서버에서 오류가 발생했습니다."}), 500

    db.session.delete(video_to_delete)
    db.session.commit()
    return jsonify({"message": "비디오가 성공적으로 삭제되었습니다."})

# [추가] 관리자 전용 API - 사용자 삭제
@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    identity_json = get_jwt_identity()
    current_user = json.loads(identity_json)

    # 1. 관리자 권한 확인
    if current_user.get('role') != 'admin':
        return jsonify({"msg": "관리자 권한이 필요합니다."}), 403

    # 2. 자기 자신을 삭제하려는지 확인
    if current_user.get('id') == user_id:
        return jsonify({"error": "자기 자신을 삭제할 수 없습니다."}), 400

    user_to_delete = User.query.get(user_id)

    if not user_to_delete:
        return jsonify({"error": "사용자를 찾을 수 없습니다."}), 404

    # 3. 다른 관리자를 삭제하려는지 확인 (선택적이지만 좋은 정책)
    if user_to_delete.role == 'admin':
        return jsonify({"error": "다른 관리자를 삭제할 수 없습니다."}), 403

    # 4. 사용자와 관련된 모든 비디오 및 파일 삭제
    videos_to_delete = Video.query.filter_by(user_id=user_id).all()
    for video in videos_to_delete:
        try:
            master_path = os.path.join(app.config['OUTPUT_FOLDER'], video.master_filename)
            playback_path = os.path.join(app.config['OUTPUT_FOLDER'], video.playback_filename)
            if os.path.exists(master_path): os.remove(master_path)
            if os.path.exists(playback_path): os.remove(playback_path)
        except Exception as e:
            app.logger.error(f"사용자 삭제 중 파일 삭제 오류 (video_id: {video.id}): {e}")
            # 파일 삭제에 실패해도 DB 삭제는 계속 진행

    # 5. 사용자 및 관련 비디오 DB 기록 삭제 (SQLAlchemy의 cascade를 이용)
    # Video 모델의 user 관계에 cascade='all, delete-orphan'을 추가하면 user만 지워도 video가 지워지지만,
    # 파일 삭제를 위해 명시적으로 처리하는 것이 더 안전합니다.
    db.session.delete(user_to_delete)
    db.session.commit()

    return jsonify({"message": f"사용자 '{user_to_delete.username}' 및 관련 데이터가 모두 삭제되었습니다."})

# [추가] 사용자 본인 회원 탈퇴 API
@app.route('/api/profile/delete-account', methods=['POST'])
@jwt_required()
def delete_account():
    identity_json = get_jwt_identity()
    current_user_data = json.loads(identity_json)
    user_id = current_user_data['id']

    data = request.get_json()
    password = data.get('password')

    if not password:
        return jsonify({'error': '비밀번호를 입력해주세요.'}), 400

    user_to_delete = User.query.get(user_id)

    if not user_to_delete or not bcrypt.check_password_hash(user_to_delete.password, password):
        return jsonify({'error': '비밀번호가 일치하지 않습니다.'}), 401

    # 사용자와 관련된 모든 비디오 및 파일 삭제
    videos_to_delete = Video.query.filter_by(user_id=user_id).all()
    for video in videos_to_delete:
        try:
            master_path = os.path.join(app.config['OUTPUT_FOLDER'], video.master_filename)
            playback_path = os.path.join(app.config['OUTPUT_FOLDER'], video.playback_filename)
            if os.path.exists(master_path): os.remove(master_path)
            if os.path.exists(playback_path): os.remove(playback_path)
        except Exception as e:
            app.logger.error(f"파일 삭제 중 오류 발생 (video_id: {video.id}): {e}")

    db.session.delete(user_to_delete)
    db.session.commit()
    return jsonify({'message': '회원 탈퇴가 성공적으로 처리되었습니다.'})

@app.route('/outputs/<filename>')
def get_output_file(filename):
    # [수정] MP4 파일은 스트리밍을 위해 인라인으로 제공, 그 외에는 다운로드
    image_extensions = ['.jpeg', '.jpg', '.png', '.gif', '.webp']
    is_image = any(filename.lower().endswith(ext) for ext in image_extensions)
    if filename.endswith('.mp4') or is_image:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=False)
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
