-- AlterTable: store Session.shop (text) in user_id instead of Shopify user ID (bigint)
ALTER TABLE "shorts" ALTER COLUMN "user_id" TYPE TEXT USING (user_id::text);
