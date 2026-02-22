-- CreateTable
CREATE TABLE "Session" (
    "id" TEXT NOT NULL,
    "shop" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "isOnline" BOOLEAN NOT NULL DEFAULT false,
    "scope" TEXT,
    "expires" TIMESTAMP(3),
    "accessToken" TEXT NOT NULL,
    "userId" BIGINT,
    "firstName" TEXT,
    "lastName" TEXT,
    "email" TEXT,
    "accountOwner" BOOLEAN NOT NULL DEFAULT false,
    "locale" TEXT,
    "collaborator" BOOLEAN DEFAULT false,
    "emailVerified" BOOLEAN DEFAULT false,
    "refreshToken" TEXT,
    "refreshTokenExpires" TIMESTAMP(3),

    CONSTRAINT "Session_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "shorts" (
    "id" TEXT NOT NULL,
    "user_id" BIGINT,
    "title" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'draft',
    "final_video_url" TEXT,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "shorts_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "video_scenes" (
    "id" TEXT NOT NULL,
    "short_id" TEXT NOT NULL,
    "scene_number" INTEGER NOT NULL,
    "duration" DOUBLE PRECISION NOT NULL,
    "image_url" TEXT,
    "generated_video_url" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "metadata" JSONB,
    "generate_video_prompt" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "video_scenes_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "audio_info" (
    "id" TEXT NOT NULL,
    "user_id" BIGINT,
    "short_id" TEXT NOT NULL,
    "voice_id" TEXT,
    "voice_name" TEXT,
    "speed" DOUBLE PRECISION DEFAULT 1,
    "volume" DOUBLE PRECISION DEFAULT 1,
    "generated_audio_url" TEXT,
    "subtitles" JSONB,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "audio_script" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "audio_info_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "promo_workflow_temp" (
    "id" TEXT NOT NULL,
    "shop" TEXT NOT NULL,
    "product_id" TEXT NOT NULL,
    "state" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "promo_workflow_temp_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "audio_info_short_id_key" ON "audio_info"("short_id");

-- CreateIndex
CREATE UNIQUE INDEX "promo_workflow_temp_shop_product_id_key" ON "promo_workflow_temp"("shop", "product_id");

-- AddForeignKey
ALTER TABLE "video_scenes" ADD CONSTRAINT "video_scenes_short_id_fkey" FOREIGN KEY ("short_id") REFERENCES "shorts"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "audio_info" ADD CONSTRAINT "audio_info_short_id_fkey" FOREIGN KEY ("short_id") REFERENCES "shorts"("id") ON DELETE CASCADE ON UPDATE CASCADE;
