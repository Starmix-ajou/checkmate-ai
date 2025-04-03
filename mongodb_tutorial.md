# MongoDB 튜토리얼 명령어

## 기본 명령어

### 데이터베이스 관련
```javascript
// 데이터베이스 목록 확인
show dbs

// 현재 데이터베이스 확인
db

// 데이터베이스 선택
use checkmate

// 데이터베이스 통계 확인
db.stats()
```

### 컬렉션 관련
```javascript
// 컬렉션 목록 확인
show collections

// 컬렉션 통계 확인
db.feature_definitions.stats()
```

### 문서 조회
```javascript
// 모든 문서 조회
db.feature_definitions.find()

// 보기 좋게 포맷팅하여 조회
db.feature_definitions.find().pretty()

// 특정 조건으로 데이터 조회
db.feature_definitions.find({"status": "draft"})

// 문서 개수 확인
db.feature_definitions.countDocuments()
```

### 정렬 및 필터링
```javascript
// 특정 필드로 정렬하여 조회
db.feature_definitions.find().sort({"status": 1})  // status 필드로 오름차순 정렬
```

### 삭제 명령어
```javascript
// 문서 삭제
db.feature_definitions.deleteOne({"_id": ObjectId("문서ID")})

// 컬렉션 삭제
db.feature_definitions.drop()

// 데이터베이스 삭제
db.dropDatabase()
```

### 종료
```javascript
// MongoDB Shell 종료
exit
```

## 실제 사용 예시
```bash
# 1. MongoDB Shell 접속
mongosh

# 2. checkmate 데이터베이스 선택
use checkmate

# 3. 컬렉션 목록 확인
show collections

# 4. feature_definitions 컬렉션의 모든 문서 조회
db.feature_definitions.find().pretty()

# 5. feature_specifications 컬렉션의 모든 문서 조회
db.feature_specifications.find().pretty()
``` 