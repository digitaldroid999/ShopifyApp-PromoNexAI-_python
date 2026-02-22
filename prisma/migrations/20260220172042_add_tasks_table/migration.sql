-- CreateTable
CREATE TABLE "tasks" (
    "id" TEXT NOT NULL,
    "remotion_task_id" TEXT NOT NULL,
    "short_id" TEXT NOT NULL,
    "video_scene_id" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "stage" TEXT,
    "progress" INTEGER DEFAULT 0,
    "video_url" TEXT,
    "error" TEXT,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "tasks_pkey" PRIMARY KEY ("id")
);
